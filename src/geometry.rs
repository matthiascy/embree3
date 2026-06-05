//! Callback closures and user-data are mutated only through a `&mut
//! GeometryBuilder`, which is the *sole* `Arc<GeometryShared>` owner (`!Clone`,
//! `!Sync`, `strong_count == 1`). A geometry cannot be a builder while attached
//! to a scene or cloned (`try_edit` needs sole ownership). Therefore every
//! shared observer (for example other clones, `Scene::attach_geometry`'s
//! retained clone, and Embree's traversal threads) sees the state only *after*
//! it is frozen. Two distinct facts make this sound:
//!
//! 1. `Arc::get_mut == Some` proves **exclusivity**: while a write happens, no
//!    `&GeometryShared` reader exists at all
//! 2. the writes become **visible** to a later reader on another thread through
//!    the ordinary happens-before of the ownership move / thread handoff that
//!    separates them; entering Embree's parallel `commit`/traversal, or an
//!    `Arc` clone sent to another application thread.
//!
//! The refcount alone does *not* publish non-atomic `UnsafeCell` bytes; the
//! handoff does. The trampolines and `callback_data` perform plain reads with
//! no lock because, by (1)+(2), no write is ever concurrent with, or
//! unpublished before them.

use std::{
    any::Any, cell::UnsafeCell, collections::HashMap, marker::PhantomData, ptr, sync::Mutex,
};

use crate::{
    buffer::required_layout_bytes, callback::ErasedFn, sys::*, AsIntersectContext, Bounds, Buffer,
    BufferData, BufferLayout, BufferSize, BufferSource, BufferUsage, BufferView, BufferViewMut,
    BuildQuality, Device, Error, Format, GeometryKind, Hit, HitN, QuaternionDecomposition, Ray,
    RayN, Scene, SoAHit, SoARay, SubdivisionMode, UserData,
};

use std::{
    borrow::Cow,
    ops::{Bound, Deref, DerefMut, Index, IndexMut, RangeBounds},
    os::raw::c_void,
    sync::Arc,
};

/// How a buffer is bound to a geometry slot (internal record; the public query
/// form is [`BufferSource`]). `Managed` owns a retained [`Buffer`] (no `'buf`
/// constraint); `Shared` borrows the caller's host bytes (ties `'buf`); `Local`
/// is embree-owned.
#[derive(Debug)]
pub(crate) enum AttachedBuffer<'buf> {
    Managed {
        buffer: Buffer,
        byte_offset: usize,
        layout: BufferLayout,
    },
    Shared {
        data: &'buf [u8],
        layout: BufferLayout,
    },
    Local {
        ptr: *mut c_void,
        size: BufferSize,
        layout: BufferLayout,
    },
}

/// Identifies one of a geometry's callback slots.
///
/// Pass it to [`GeometryBuilder::callback_data`] /
/// [`Geometry::callback_data`] to read the owned data bound to a specific
/// callback. Internally it is also the **single source of truth** for slot
/// ordering: the (private) `CallSite` and `CallbackOwners` tables are arrays
/// indexed by `kind as usize`, so there is no parallel name↔index mapping to
/// drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CbKind {
    IntersectFilter,
    OccludedFilter,
    UserIntersect,
    UserOccluded,
    UserBounds,
    Displacement,
}

impl CbKind {
    /// Number of callback kinds used for the length of the per-geometry
    /// callback slot arrays.
    pub const COUNT: usize = 6;
}

/// One callback slot, read by the trampoline on every invocation: a raw pointer
/// to the boxed closure plus *that callback's own* data pointer (null = no data
/// bound). Closure and data are bound together at the setter with a
/// statically-known `D`, so the trampoline which monomorphized over that same
/// `D`, casts `user_data` **unchecked**; there is no resolution table and no
/// `TypeId` on the hot path. The pointers are raw because the writer (the
/// unique `&mut GeometryBuilder`) and the reader (the embree trampoline) never
/// overlap (see the module invariant).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct Slot {
    pub closure: *const (), // -> the boxed `F`; recovered as `&F` (monomorphized)
    pub user_data: *const (), // -> the callback's own `*const D`, or null
}

impl Slot {
    const EMPTY: Slot = Slot {
        closure: ptr::null(),
        user_data: ptr::null(),
    };
}

/// Hot, immutable-after-publish callback table that embree's `geometryUserPtr`
/// points directly at, so a trampoline resolves its closure + data with a
/// single indexed read and no pointer-chasing. `#[repr(C, align(64))]` puts it
/// on its own cache line(s), away from the cold `owners`/`attachments` that
/// share the enclosing `Arc` allocation, so concurrent reads from many
/// traversal threads never false-share with a builder's cold writes.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct CallSite {
    pub slots: [Slot; CbKind::COUNT],
}

impl CallSite {
    const EMPTY: CallSite = CallSite {
        slots: [Slot::EMPTY; CbKind::COUNT],
    };
}

/// Cold side of the callback table: keeps each slot's boxed closure alive (the
/// `Slot.closure` raw pointer borrows from it) and, for *owned* data, the
/// `Box<dyn Any>` whose `Slot.user_data` points into. Borrowed data has no
/// entry here as the application owns it, kept valid by the geometry's `'buf`
/// lifetime. Indexed by `CbKind as usize`. Touched only by the builder setters
/// and [`callback_data`](GeometryBuilder::callback_data); never by a
/// trampoline.
#[derive(Debug, Default)]
pub(crate) struct CallbackOwners {
    pub closures: [Option<ErasedFn>; CbKind::COUNT],
    pub owned_data: [Option<Box<dyn Any + Send + Sync>>; CbKind::COUNT],
}

/// Heap-owned callback state that embree's `geometryUserPtr` points into: the
/// hot [`CallSite`] (read by trampolines) and the cold [`CallbackOwners`] (the
/// closures/data they reference). Both are `UnsafeCell` because they are
/// mutated through a `&GeometryShared` (the builder writes via the shared
/// `Arc`) yet read lock-free; soundness rests on the module invariant(see
/// [`GeometryShared`]).
#[derive(Debug)]
pub(crate) struct GeometryData {
    pub call_site: UnsafeCell<CallSite>,
    pub owners: UnsafeCell<CallbackOwners>,
}

impl Default for GeometryData {
    fn default() -> Self {
        Self {
            call_site: UnsafeCell::new(CallSite::EMPTY),
            owners: UnsafeCell::new(CallbackOwners::default()),
        }
    }
}

/// All per-geometry state, heap-owned for the object's whole lifetime.
///
/// Its address is embree's `userPtr` (set in `new`), so it must never move.
#[derive(Debug)]
pub(crate) struct GeometryShared<'buf> {
    pub(crate) device: Device,
    pub(crate) handle: RTCGeometry,
    pub(crate) kind: GeometryKind,
    pub(crate) attachments: Mutex<HashMap<(BufferUsage, u32), AttachedBuffer<'buf>>>,
    pub(crate) data: GeometryData,
}

impl<'buf> Drop for GeometryShared<'buf> {
    fn drop(&mut self) {
        // Released exactly once, when the last wrapper (and the scene's clone)
        // is gone.
        unsafe {
            rtcReleaseGeometry(self.handle);
        }
    }
}

// SAFETY: `GeometryData`'s `UnsafeCell`s are mutated only through a `&mut
// GeometryBuilder`, the unique `Arc<GeometryShared>` owner (strong_count == 1,
// !Clone, !Sync). No shared observer (other clones, the scene's retained clone,
// Embree's traversal threads) coexists with that mutation; and the ownership
// move / thread handoff that must separate the last write from any later shared
// read is what makes the writes visible (see the module invariant). So a shared
// `&Geometry` only ever observes frozen state. The boxed `F`/`D` are
// `Send + Sync` (enforced at registration).
unsafe impl Sync for GeometryShared<'_> {}

impl<'buf> GeometryShared<'buf> {
    /// Snapshot of the buffer bound at `(usage, slot)`, owning/copying out of
    /// the lock guard (a `Managed` retain clone, the `Shared` host borrow,
    /// or `Local` metadata) so the result does not borrow the `attachments`
    /// lock.
    fn buffer_source(&self, usage: BufferUsage, slot: u32) -> Option<BufferSource<'buf>> {
        let attachments = self.attachments.lock().unwrap();
        attachments.get(&(usage, slot)).map(|a| match a {
            AttachedBuffer::Managed {
                buffer,
                byte_offset,
                layout,
            } => BufferSource::Managed {
                buffer: buffer.clone(),
                byte_offset: *byte_offset,
                layout: *layout,
            },
            AttachedBuffer::Shared { data, layout } => BufferSource::Shared {
                data,
                layout: *layout,
            },
            AttachedBuffer::Local { size, layout, .. } => BufferSource::Local {
                size: *size,
                layout: *layout,
            },
        })
    }

    /// Resolves a geometry-**local** buffer slot to `(ptr, element_count)` for
    /// mapping it as `[T]`, with the runtime layout checks (`T` tiles the
    /// allocation, `T`-aligned pointer, non-ZST). `Err(INVALID_ARGUMENT)`
    /// if the slot is unbound or not a local buffer, or the checks fail.
    /// Only `Local` buffers are mappable this way: `Managed`
    /// is mapped through its `Buffer`, `Shared` is the caller's own slice.
    pub(crate) fn map_local<T: BufferData>(
        &self,
        usage: BufferUsage,
        slot: u32,
    ) -> Result<(*mut T, usize), Error> {
        let byte_size = {
            let attachments = self.attachments.lock().unwrap();
            match attachments.get(&(usage, slot)) {
                Some(AttachedBuffer::Local { size, .. }) => size.get(),
                _ => return Err(Error::INVALID_ARGUMENT),
            }
        };
        let ptr = unsafe { rtcGetGeometryBufferData(self.handle, usage, slot) } as *mut T;
        let t_size = std::mem::size_of::<T>();
        if ptr.is_null()
            || t_size == 0
            || byte_size % t_size != 0
            || (ptr as usize) % std::mem::align_of::<T>() != 0
        {
            return Err(Error::INVALID_ARGUMENT);
        }
        Ok((ptr, byte_size / t_size))
    }

    /// SAFETY: caller has exclusive access (unique `&mut GeometryBuilder`).
    unsafe fn data_mut(&self) -> (&mut CallSite, &mut CallbackOwners) {
        (
            &mut *self.data.call_site.get(),
            &mut *self.data.owners.get(),
        )
    }
}

/// The **mutable build/edit phase** of an Embree geometry.
///
/// A geometry starts here (created via [`Device::create_geometry`]), the typed
/// constructors (`TriangleMesh::new`, `QuadMesh::new`, ...), or
/// [`Geometry::new`]. You configure it in this phase and then call
/// [`commit`](GeometryBuilder::commit) to obtain a shareable, read-only
/// [`Geometry`] that can be attached to scenes.
///
/// Depending on the geometry type, different buffers must be bound (typically
/// a vertex and an index buffer) using
/// [`set_buffer`](GeometryBuilder::set_buffer) or
/// [`set_new_buffer`](GeometryBuilder::set_new_buffer). The primitive and
/// vertex counts are usually inferred from the bound buffer sizes.
///
/// All geometry types support multi-segment motion blur with 2..=129
/// equidistant time steps inside a user-specified time range: set the count
/// with [`set_time_step_count`](GeometryBuilder::set_time_step_count), bind one
/// vertex buffer per time step, and optionally a time range (geometries may
/// also appear / disappear during the shutter if the range is a sub-range of
/// `[0, 1]`).
///
/// Per-geometry intersection / occlusion **filter** callbacks
/// ([`set_intersect_filter_function`](GeometryBuilder::set_intersect_filter_function)
/// and the occlusion counterpart) are invoked for each hit found during
/// `Scene::intersect` / `Scene::occluded` and let you discard intersections
/// (e.g. to model alpha-cutout silhouettes such as tree leaves).
///
/// # Thread safety
///
/// `GeometryBuilder` is `Send` but **not** `Sync` and **not** `Clone`: it is
/// the *unique* owner of its geometry, so `&mut self` is genuine exclusive
/// access. That is exactly embree's contract that a single geometry must be
/// modified by at most one thread at a time. You may move a builder to another
/// thread to build it there, but you cannot share it. Regain a builder from a
/// committed geometry with [`Geometry::try_edit`].
///
/// A `GeometryBuilder` cannot be cloned (that would create an aliased mutable
/// handle):
///
/// ```compile_fail
/// use embree3::{Device, GeometryKind};
/// let device = Device::new().unwrap();
/// let builder = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
/// let _alias = builder.clone(); // error: GeometryBuilder is not Clone
/// ```
///
/// nor shared across threads (`!Sync`):
///
/// ```compile_fail
/// use embree3::{Device, GeometryKind};
/// fn needs_sync<T: Sync>(_: &T) {}
/// let device = Device::new().unwrap();
/// let builder = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
/// needs_sync(&builder); // error: GeometryBuilder is not Sync
/// ```
#[derive(Debug)]
pub struct GeometryBuilder<'buf> {
    pub(crate) shared: Arc<GeometryShared<'buf>>,
}

// SAFETY: a `GeometryBuilder` is the *unique* owner of its
// `Arc<GeometryShared>` (it is never `Clone`d, and `try_edit` only produces one
// when strong_count == 1). Its state is `Send` (closures are `Fn + Send +
// Sync`, user data `Send + Sync`). Moving it transfers exclusive access, which
// maps exactly to embree's "one thread modifies one geometry". It is
// intentionally NOT `Sync` and NOT `Clone`.
unsafe impl<'buf> Send for GeometryBuilder<'buf> {}

impl<'buf> GeometryBuilder<'buf> {
    /// Binds caller-owned host memory as a geometry buffer, zero-copy
    /// (`rtcSetSharedGeometryBuffer`). `data` must outlive the geometry; the
    /// `'buf` borrow enforces it.
    ///
    /// Embree reads `data[0 .. (count-1)*stride + tail]`, where `tail` is the
    /// element size, **rounded up to 16 bytes for a vertex buffer** (embree
    /// SSE-reads the last element). The host pointer and `stride` must be
    /// 4-byte aligned. Pre-slice `data` for a non-zero start, there is no
    /// separate byte offset.
    pub fn set_shared_buffer(
        &mut self,
        usage: BufferUsage,
        slot: u32,
        format: Format,
        data: &'buf [u8],
        stride: usize,
        count: usize,
    ) -> Result<(), Error> {
        if usage == BufferUsage::VERTEX_ATTRIBUTE {
            self.check_vertex_attribute()?;
        }
        let vertex = matches!(usage, BufferUsage::VERTEX | BufferUsage::VERTEX_ATTRIBUTE);
        let req =
            required_layout_bytes(format, stride, count, vertex).ok_or(Error::INVALID_ARGUMENT)?;
        if data.len() < req || (data.as_ptr() as usize) % 4 != 0 {
            return Err(Error::INVALID_ARGUMENT);
        }
        unsafe {
            rtcSetSharedGeometryBuffer(
                self.shared.handle,
                usage,
                slot,
                format,
                data.as_ptr() as *const c_void,
                0, // caller pre-slices; embree gets no separate byteOffset
                stride,
                count,
            );
        }
        let layout = BufferLayout {
            format,
            stride,
            count,
        };
        self.shared
            .attachments
            .lock()
            .unwrap()
            .insert((usage, slot), AttachedBuffer::Shared { data, layout });
        Ok(())
    }

    /// Binds a byte sub-range of a refcounted [`Buffer`]
    /// (`rtcSetGeometryBuffer`). The geometry **retains** the buffer, so
    /// this does not constrain the geometry's lifetime. `byte_range`'s
    /// start is the byte offset (must be 4-byte aligned); the range must
    /// lie within the buffer and be long enough for the layout.
    pub fn set_managed_buffer<S: RangeBounds<usize>>(
        &mut self,
        usage: BufferUsage,
        slot: u32,
        format: Format,
        buffer: &Buffer,
        byte_range: S,
        stride: usize,
        count: usize,
    ) -> Result<(), Error> {
        if usage == BufferUsage::VERTEX_ATTRIBUTE {
            self.check_vertex_attribute()?;
        }
        let byte_offset = match byte_range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };
        let end = match byte_range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => buffer.size.get(),
        };
        let vertex = matches!(usage, BufferUsage::VERTEX | BufferUsage::VERTEX_ATTRIBUTE);
        let req =
            required_layout_bytes(format, stride, count, vertex).ok_or(Error::INVALID_ARGUMENT)?;
        if byte_offset % 4 != 0
            || byte_offset > end
            || end > buffer.size.get()
            || end - byte_offset < req
        {
            return Err(Error::INVALID_ARGUMENT);
        }
        unsafe {
            rtcSetGeometryBuffer(
                self.shared.handle,
                usage,
                slot,
                format,
                buffer.handle,
                byte_offset,
                stride,
                count,
            );
        }
        let layout = BufferLayout {
            format,
            stride,
            count,
        };
        self.shared.attachments.lock().unwrap().insert(
            (usage, slot),
            AttachedBuffer::Managed {
                buffer: buffer.clone(), // rtcRetainBuffer
                byte_offset,
                layout,
            },
        );
        Ok(())
    }

    /// Creates a new [`Buffer`](`crate::Buffer`) and binds it as a specific
    /// attribute for this geometry.
    ///
    /// Analogous to [`rtcSetNewGeometryBuffer`](https://spec.oneapi.io/oneart/0.5-rev-1/embree-spec.html#rtcsetnewgeometrybuffer).
    ///
    /// The allocated buffer will be automatically over-allocated slightly when
    /// used as a [`BufferUsage::VERTEX`] buffer, where a requirement is
    /// that each buffer element should be readable using 16-byte SSE load
    /// instructions. This means that the buffer will be padded to a multiple of
    /// 16 bytes.
    ///
    /// The allocated buffer is managed internally and automatically released
    /// when the geometry is destroyed by Embree.
    ///
    /// # Arguments
    ///
    /// * `usage` - The usage of the buffer.
    ///
    /// * `slot` - The slot to bind the buffer to.
    ///
    /// * `format` - The format of the buffer items. See [`Format`] for more
    ///   information.
    ///
    /// * `count` - The number of items in the buffer.
    ///
    /// * `stride` - The stride of the buffer items. MUST be a multiple of 4.
    pub fn set_new_buffer<T: BufferData>(
        &mut self,
        usage: BufferUsage,
        slot: u32,
        format: Format,
        stride: usize,
        count: usize,
    ) -> Result<BufferViewMut<'_, T>, Error> {
        if usage == BufferUsage::VERTEX_ATTRIBUTE {
            self.check_vertex_attribute()?;
        }
        let vertex = matches!(usage, BufferUsage::VERTEX | BufferUsage::VERTEX_ATTRIBUTE);
        // Validates count >= 1, known format, stride >= elem & 4-aligned, no overflow.
        required_layout_bytes(format, stride, count, vertex).ok_or(Error::INVALID_ARGUMENT)?;
        let size = stride.checked_mul(count).ok_or(Error::INVALID_ARGUMENT)?;
        let t_size = std::mem::size_of::<T>();
        if t_size == 0 || size % t_size != 0 {
            return Err(Error::INVALID_ARGUMENT);
        }
        let raw_ptr = unsafe {
            rtcSetNewGeometryBuffer(self.shared.handle, usage, slot, format, stride, count)
        };
        if raw_ptr.is_null() {
            return Err(self.shared.device.get_error());
        }
        if (raw_ptr as usize) % std::mem::align_of::<T>() != 0 {
            return Err(Error::INVALID_ARGUMENT);
        }
        let layout = BufferLayout {
            format,
            stride,
            count,
        };
        self.shared.attachments.lock().unwrap().insert(
            (usage, slot),
            AttachedBuffer::Local {
                ptr: raw_ptr,
                size: BufferSize::new(size).ok_or(Error::INVALID_ARGUMENT)?,
                layout,
            },
        );
        // SAFETY: embree-allocated storage of `size` bytes; `T: BufferData` tiles it
        // (`size % t_size == 0`), the pointer is `T`-aligned, and `&mut self` (the
        // unique builder) gives exclusive access for the returned view's borrow.
        Ok(unsafe { BufferViewMut::from_raw_parts(raw_ptr as *mut T, size / t_size) })
    }

    /// Marks a buffer slice bound to this geometry as modified.
    ///
    /// If a data buffer is changed by the application, this function must be
    /// called for the buffer to be updated in the geometry. Each buffer slice
    /// assigned to a buffer slot is initially marked as modified, thus this
    /// method needs to be called only when doing buffer modifications after the
    /// first [`Scene::commit`] call.
    pub fn update_buffer(&mut self, usage: BufferUsage, slot: u32) {
        unsafe {
            rtcUpdateGeometryBuffer(self.shared.handle, usage, slot);
        }
    }

    /// Disables the geometry, so it is not rendered. Each geometry is enabled
    /// by default at construction time.
    ///
    /// This modifies the geometry, so it lives on the builder (the build/edit
    /// phase). To toggle a geometry that is already attached to a scene during
    /// a render loop, use
    /// [`Scene::disable_geometry`](crate::Scene::disable_geometry)
    /// instead (it excludes concurrent traversal via `&mut Scene`). After the
    /// change, the containing scene must be committed for it to take effect.
    pub fn disable(&mut self) {
        unsafe {
            rtcDisableGeometry(self.shared.handle);
        }
    }

    /// Enables the geometry, so it is rendered. Each geometry is enabled by
    /// default at construction time.
    ///
    /// See [`GeometryBuilder::disable`] for the build-phase vs. dynamic
    /// ([`Scene::enable_geometry`](crate::Scene::enable_geometry)) distinction.
    /// After the change, the containing scene must be committed for it to take
    /// effect.
    pub fn enable(&mut self) {
        unsafe {
            rtcEnableGeometry(self.shared.handle);
        }
    }

    /// Sets the number of vertex attributes of the geometry.
    ///
    /// This function sets the number of slots for vertex attributes buffers
    /// (BufferUsage::VERTEX_ATTRIBUTE) that can be used for the specified
    /// geometry.
    ///
    /// Only supported by triangle meshes, quad meshes, curves, points, and
    /// subdivision geometries.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of vertex attribute slots.
    pub fn set_vertex_attribute_count(&mut self, count: u32) {
        match self.shared.kind {
            // Vertex attributes are not supported by these kinds; no-op.
            GeometryKind::GRID | GeometryKind::USER | GeometryKind::INSTANCE => {}
            _ => {
                // Update the vertex attribute count.
                unsafe {
                    rtcSetGeometryVertexAttributeCount(self.shared.handle, count);
                }
            }
        }
    }

    /// Sets the build quality for the geometry.
    ///
    /// The per-geometry build quality is only a hint and may be ignored. Embree
    /// currently uses the per-geometry build quality when the scene build
    /// quality is set to [`BuildQuality::LOW`]. In this mode a two-level
    /// acceleration structure is build, and geometries build a separate
    /// acceleration structure using the geometry build quality.
    ///
    /// The build quality can be one of the following:
    ///
    /// - [`BuildQuality::LOW`]: Creates lower quality data structures, e.g. for
    ///   dynamic scenes.
    ///
    /// - [`BuildQuality::MEDIUM`]: Default build quality for most usages. Gives
    ///   a good balance between quality and performance.
    ///
    /// - [`BuildQuality::HIGH`]: Creates higher quality data structures for
    ///   final frame rendering. Enables a spatial split builder for certain
    ///   primitive types.
    ///
    /// - [`BuildQuality::REFIT`]: Uses a BVH refitting approach when changing
    ///   only the vertex buffer.
    pub fn set_build_quality(&mut self, quality: BuildQuality) {
        unsafe {
            rtcSetGeometryBuildQuality(self.shared.handle, quality);
        }
    }

    /// Sets the tessellation rate for a subdivision mesh or flat curves.
    ///
    /// For curves, the tessellation rate specifies the number of ray-facing
    /// quads per curve segment. For subdivision surfaces, the tessellation
    /// rate specifies the number of quads along each edge.
    pub fn set_tessellation_rate(&mut self, rate: f32) {
        match self.shared.kind {
            GeometryKind::SUBDIVISION
            | GeometryKind::FLAT_LINEAR_CURVE
            | GeometryKind::FLAT_BEZIER_CURVE
            | GeometryKind::ROUND_LINEAR_CURVE
            | GeometryKind::ROUND_BEZIER_CURVE => unsafe {
                rtcSetGeometryTessellationRate(self.shared.handle, rate);
            },
            _ => panic!(
                "GeometryBuilder::set_tessellation_rate is only supported for subdivision meshes \
                 and flat curves"
            ),
        }
    }

    /// Sets the mask for the geometry.
    ///
    /// This geometry mask is used together with the ray mask stored inside the
    /// mask field of the ray. The primitives of the geometry are hit by the ray
    /// only if the bitwise and operation of the geometry mask with the ray mask
    /// is not 0.
    /// This feature can be used to disable selected geometries for specifically
    /// tagged rays, e.g. to disable shadow casting for certain geometries.
    ///
    /// Ray masks are disabled in Embree by default at compile time, and can be
    /// enabled through the `EMBREE_RAY_MASK` parameter in CMake. One can query
    /// whether ray masks are enabled by querying the
    /// [`DeviceProperty::RAY_MASK_SUPPORTED`](`crate::DeviceProperty::RAY_MASK_SUPPORTED`)
    /// device property using [`Device::get_property`].
    pub fn set_mask(&mut self, mask: u32) {
        unsafe {
            rtcSetGeometryMask(self.shared.handle, mask);
        }
    }

    /// Sets the number of time steps for multi-segment motion blur for the
    /// geometry.
    ///
    /// For triangle meshes, quad meshes, curves, points, and subdivision
    /// geometries, the number of time steps directly corresponds to the
    /// number of vertex buffer slots available [`BufferUsage::VERTEX`].
    ///
    /// For instance geometries, a transformation must be specified for each
    /// time step (see [`GeometryBuilder::set_transform`]).
    ///
    /// For user geometries, the registered bounding callback function must
    /// provide a bounding box per primitive and time step, and the
    /// intersection and occlusion callback functions should properly
    /// intersect the motion-blurred geometry at the ray time.
    pub fn set_time_step_count(&mut self, count: u32) {
        unsafe {
            rtcSetGeometryTimeStepCount(self.shared.handle, count);
        }
    }

    /// Sets the time range for a motion blur geometry.
    ///
    /// The time range is defined relative to the camera shutter interval
    /// \[0,1\] but it can be arbitrary. Thus the `start` time can be
    /// smaller, equal, or larger 0, indicating a geometry whose animation
    /// definition start before, at, or after the camera shutter opens.
    /// Similar the `end` time can be smaller, equal, or larger than 1,
    /// indicating a geometry whose animation definition ends after, at, or
    /// before the camera shutter closes. The `start` time has to be smaller
    /// or equal to the `end` time.
    ///
    /// The default time range when this function is not called is the entire
    /// camera shutter \[0,1\]. For best performance at most one time segment
    /// of the piece wise linear definition of the motion should fall
    /// outside the shutter window to the left and to the right. Thus do not
    /// set the `start` time or `end` time too far outside the
    /// \[0,1\] interval for best performance.
    ///
    /// This time range feature will also allow geometries to appear and
    /// disappear during the camera shutter time if the specified time range
    /// is a sub range of \[0,1\].
    ///
    /// Please also have a look at the [`GeometryBuilder::set_time_step_count`]
    /// to see how to define the time steps for the specified time range.
    pub fn set_time_range(&mut self, start: f32, end: f32) {
        unsafe {
            rtcSetGeometryTimeRange(self.shared.handle, start, end);
        }
    }

    /// Registers an intersection filter callback function for the geometry.
    ///
    /// Only a single callback function can be registered per geometry, and
    /// further invocations overwrite the previously set callback function.
    /// Unregister the callback function by calling
    /// [`GeometryBuilder::unset_intersect_filter_function`].
    ///
    /// The registered filter function is invoked for every hit encountered
    /// during the intersect-type ray queries and can accept or reject that
    /// hit. The feature can be used to define a silhouette for a primitive
    /// and reject hits that are outside the silhouette. E.g. a tree leaf
    /// could be modeled with an alpha texture that decides whether hit
    /// points lie inside or outside the leaf.
    ///
    /// If [`BuildQuality::HIGH`] is set, the filter functions may be called
    /// multiple times for the same primitive hit. Further, rays hitting
    /// exactly the edge might also report two hits for the same surface. For
    /// certain use cases, the application may have to work around this
    /// limitation by collecting already reported hits (geomID/primID pairs)
    /// and ignoring duplicates.
    ///
    /// The filter function callback of type [`RTCFilterFunctionN`] gets passed
    /// a number of arguments through the [`RTCFilterFunctionNArguments`]
    /// structure. The valid parameter of that structure points to an
    /// integer valid mask (0 means invalid and -1 means valid). The
    /// `geometryUserPtr` member is handled by the wrapper: the data optionally
    /// bound to *this* callback (via its `_owned` / `_borrowed` variant) is
    /// delivered as the closure's `Option<&D>` argument. The
    /// context member points to the intersection context passed to
    /// the ray query function. The ray parameter points to N rays in SOA layout
    /// (see `RayN`, `HitN`).
    /// The hit parameter points to N hits in SOA layout to test. The N
    /// parameter is the number of rays and hits in ray and hit. The hit
    /// distance is provided as the tfar value of the ray. If the hit
    /// geometry is instanced, the `instID` member of the ray is valid, and
    /// the ray and the potential hit are in object space.
    ///
    /// The filter callback function has the task to check for each valid ray
    /// whether it wants to accept or reject the corresponding hit. To
    /// reject a hit, the filter callback function just has to *write 0* to
    /// the integer valid mask of the corresponding ray. To accept the hit,
    /// it just has to *leave the valid mask set to -1*. The filter function
    /// is further allowed to change the hit and decrease the tfar value of the
    /// ray but it should not modify other ray data nor any inactive
    /// components of the ray or hit.
    ///
    /// When performing ray queries using [`Scene::intersect`], it is
    /// *guaranteed* that the packet size is 1 when the callback is invoked.
    /// When performing ray queries using the [`Scene::intersect4/8/16`]
    /// functions, it is not generally guaranteed that the ray packet size
    /// (and order of rays inside the packet) passed to the callback matches
    /// the initial ray packet. However, under some circumstances these
    /// properties are guaranteed, and whether this is the case can be
    /// queried using [`Device::get_property`]. When performing ray queries
    /// using the stream API such as [`Scene::intersect_stream_aos`],
    /// [`Scene::intersect_stream_soa`], the order of rays and ray packet size
    /// of the callback function might change to either 1, 4, 8, or 16.
    ///
    /// For many usage scenarios, repacking and re-ordering of rays does not
    /// cause difficulties in implementing the callback function. However,
    /// algorithms that need to extend the ray with additional data must use
    /// the rayID component of the ray to identify the original ray to
    /// access the per-ray data.
    ///
    /// # Thread safety
    ///
    /// Embree may invoke this callback from multiple threads concurrently, for
    /// example during a parallel [`Scene::commit`](crate::Scene::commit),
    /// or when ray queries are issued from several threads on a shared
    /// scene. The closure must therefore be safe to call from several
    /// threads at once and to share across them: it must not depend
    /// on exclusive `&mut` access to its captures, and everything it captures
    /// must be `Send + Sync`. The `Fn + Send + Sync` bounds on the closure
    /// enforce this.
    pub fn set_intersect_filter_function<F, D, C>(&mut self, filter: F)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        // Register the trampoline first, then store the owner (so the old one, if any,
        // is dropped only after the new closure is installed).
        unsafe {
            rtcSetGeometryIntersectFilterFunction(
                self.shared.handle,
                trampoline::intersect_filter_function::<F, D, C>(),
            );
            self.install_callback(
                CbKind::IntersectFilter,
                ErasedFn::new(filter),
                std::ptr::null(),
                None,
            );
        }
    }

    /// Registers an intersection filter that receives **owned** per-callback
    /// data as its `Option<&D>` argument.
    ///
    /// Identical to
    /// [`set_intersect_filter_function`](Self::set_intersect_filter_function)
    /// (see it for the full filter contract and thread-safety bounds) except
    /// that this callback gets its *own* `data`, distinct from every other
    /// callback's. The geometry takes ownership of `data` and drops it exactly
    /// once, when this slot is replaced, or when the last geometry clone is
    /// dropped. Read it back outside the callback with
    /// [`callback_data`](Self::callback_data) /
    /// [`callback_data_mut`](Self::callback_data_mut).
    ///
    /// Use this when the geometry should own the data. To instead lend data you
    /// keep on the application side (zero-copy, no refcount), use
    /// [`set_intersect_filter_function_borrowed`](Self::set_intersect_filter_function_borrowed).
    pub fn set_intersect_filter_function_owned<F, D, C>(&mut self, filter: F, data: D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        // SOUNDNESS: the owned-data pointer is taken before coercing `Box<D>` to
        // `Box<dyn Any>`; coercion does not move the `D`, so `ptr` remains
        // valid for the box's lifetime.
        let boxed = Box::new(data);
        let ptr = &*boxed as *const D as *const ();
        unsafe {
            rtcSetGeometryIntersectFilterFunction(
                self.shared.handle,
                trampoline::intersect_filter_function::<F, D, C>(),
            );
            self.install_callback(
                CbKind::IntersectFilter,
                ErasedFn::new(filter),
                ptr,
                Some(boxed),
            );
        }
    }

    /// Registers an intersection filter that receives **borrowed** per-callback
    /// data as its `Option<&D>` argument.
    ///
    /// Identical to
    /// [`set_intersect_filter_function`](Self::set_intersect_filter_function)
    /// (see it for the full filter contract and thread-safety bounds) except
    /// that this callback reads `data` that the *application* owns. Nothing is
    /// allocated or reference-counted: `data` is borrowed for the geometry's
    /// lifetime `'buf` (the same lifetime that bounds shared vertex buffers)
    /// so the borrow checker forbids `data` from being dropped while the
    /// geometry (and hence any traversal that could invoke the callback) is
    /// still alive.
    ///
    /// Use this to share long-lived application data without an `Arc`. To make
    /// the geometry own the data instead, use
    /// [`set_intersect_filter_function_owned`](Self::set_intersect_filter_function_owned).
    ///
    /// The borrow is enforced at compile time, data dropped before the
    /// geometry is a type error:
    ///
    /// ```compile_fail
    /// # use embree3::{Device, GeometryKind, IntersectContext};
    /// let device = Device::new().unwrap();
    /// let mut tri = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    /// let data = vec![1u32, 2, 3];
    /// tri.set_intersect_filter_function_borrowed::<_, Vec<u32>, IntersectContext>(
    ///     |_r, _h, _v, _c, _ud| {},
    ///     &data,
    /// );
    /// drop(data); // ERROR: `data` is borrowed by `tri` for its `'buf`
    /// let _ = tri.commit(); // `tri` (holding the borrow) is still used here
    /// ```
    pub fn set_intersect_filter_function_borrowed<F, D, C>(&mut self, filter: F, data: &'buf D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        // SOUNDNESS: The borrowed form's `&'buf D` ties the data to the geometry's
        // `'buf` (the same `'buf` shared buffers already use), so the borrow checker
        // forbids the geometry (hence traversal) from outliving the data.
        // `GeometryShared<'buf>` already *uses* `'buf` (via `AttachedBuffer::Shared`);
        // keep it so (a `PhantomData<&'buf ()>` if needed) so the erased `*const ()` is
        // not silently `'static`.
        let ptr = &*data as *const D as *const ();
        unsafe {
            rtcSetGeometryIntersectFilterFunction(
                self.shared.handle,
                trampoline::intersect_filter_function::<F, D, C>(),
            );
            self.install_callback(CbKind::IntersectFilter, ErasedFn::new(filter), ptr, None);
        }
    }

    /// Unsets the intersection filter function for the geometry.
    pub fn unset_intersect_filter_function(&mut self) {
        unsafe {
            rtcSetGeometryIntersectFilterFunction(self.shared.handle, None);
            self.clear_callback(CbKind::IntersectFilter);
        }
    }

    /// Sets the occlusion filter for the geometry.
    ///
    /// Only a single callback function can be registered per geometry, and
    /// further invocations overwrite the previously set callback function.
    /// Unregister the callback function by calling
    /// [`GeometryBuilder::unset_occluded_filter_function`].
    ///
    /// The registered intersection filter function is invoked for every hit
    /// encountered during the occluded-type ray queries and can accept or
    /// reject that hit.
    ///
    /// The feature can be used to define a silhouette for a primitive and
    /// reject hits that are outside the silhouette. E.g. a tree leaf could
    /// be modeled with an alpha texture that decides whether hit points lie
    /// inside or outside the leaf. Please see the description of the
    /// [`GeometryBuilder::set_intersect_filter_function`] for a description of
    /// the filter callback function.
    ///
    /// # Thread safety
    ///
    /// Embree may invoke this callback from multiple threads concurrently, for
    /// example during a parallel [`Scene::commit`](crate::Scene::commit),
    /// or when ray queries are issued from several threads on a shared
    /// scene. The closure must therefore be safe to call from several
    /// threads at once and to share across them: it must not depend
    /// on exclusive `&mut` access to its captures, and everything it captures
    /// must be `Send + Sync`. The `Fn + Send + Sync` bounds on the closure
    /// enforce this.
    pub fn set_occluded_filter_function<F, D, C>(&mut self, filter: F)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        // Register the trampoline first, then store the owner (so the old one, if any)
        // is dropped only after the new closure is installed).
        unsafe {
            rtcSetGeometryOccludedFilterFunction(
                self.shared.handle,
                trampoline::occluded_filter_function::<F, D, C>(),
            );
            self.install_callback(
                CbKind::OccludedFilter,
                ErasedFn::new(filter),
                std::ptr::null(),
                None,
            );
        }
    }

    /// The owned-data variant of
    /// [`set_occluded_filter_function`](Self::set_occluded_filter_function).
    /// See
    /// [`set_intersect_filter_function_owned`](Self::set_intersect_filter_function_owned)
    /// for the per-callback owned-vs-borrowed data model.
    pub fn set_occluded_filter_function_owned<F, D, C>(&mut self, filter: F, data: D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        // SOUNDNESS: the owned-data pointer is taken before coercing `Box<D>` to
        // `Box<dyn Any>`; coercion does not move the `D`, so `ptr` remains
        // valid for the box's lifetime.
        let boxed = Box::new(data);
        let ptr = &*boxed as *const D as *const ();
        unsafe {
            rtcSetGeometryOccludedFilterFunction(
                self.shared.handle,
                trampoline::occluded_filter_function::<F, D, C>(),
            );
            self.install_callback(
                CbKind::OccludedFilter,
                ErasedFn::new(filter),
                ptr,
                Some(boxed),
            );
        }
    }

    /// The borrowed-data variant of
    /// [`set_occluded_filter_function`](Self::set_occluded_filter_function).
    /// See
    /// [`set_intersect_filter_function_borrowed`](Self::set_intersect_filter_function_borrowed)
    /// for the per-callback owned-vs-borrowed data model.
    pub fn set_occluded_filter_function_borrowed<F, D, C>(&mut self, filter: F, data: &'buf D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        // SOUNDNESS: The borrowed form's `&'buf D` ties the data to the geometry's
        // `'buf` (the same `'buf` shared buffers already use), so the borrow checker
        // forbids the geometry (hence traversal) from outliving the data.
        // `GeometryShared<'buf>` already *uses* `'buf` (via `AttachedBuffer::Shared`);
        // keep it so (a `PhantomData<&'buf ()>` if needed) so the erased `*const ()` is
        // not silently `'static`.
        let ptr = &*data as *const D as *const ();
        unsafe {
            rtcSetGeometryOccludedFilterFunction(
                self.shared.handle,
                trampoline::occluded_filter_function::<F, D, C>(),
            );
            self.install_callback(CbKind::OccludedFilter, ErasedFn::new(filter), ptr, None);
        }
    }

    /// Unsets the occlusion filter function for the geometry.
    pub fn unset_occluded_filter_function(&mut self) {
        unsafe {
            rtcSetGeometryOccludedFilterFunction(self.shared.handle, None);
            self.clear_callback(CbKind::OccludedFilter);
        }
    }

    // TODO(yang): how to handle the closure? RTCPointQueryFunctionArguments has a
    // user pointer but we can't set it here, instead we can only set it in the
    // rtcPointQuery function which is attached to the scene. This requires the
    // user to call [`Scene::point_query`] first and then call
    // [`GeometryBuilder::set_point_query_function`] to set the closure. Or we can
    // make the closure a member of the [`GeometryData`] and set it here.

    /// Sets the point query callback function for a geometry.
    ///
    /// Only a single callback function can be registered per geometry and
    /// further invocations overwrite the previously set callback function.
    /// Unregister the callback function by calling
    /// [`GeometryBuilder::unset_point_query_function`].
    ///
    /// The registered callback function is invoked by rtcPointQuery for every
    /// primitive of the geometry that intersects the corresponding point query
    /// domain. The callback function of type `RTCPointQueryFunction` gets
    /// passed a number of arguments through the
    /// `RTCPointQueryFunctionArguments` structure. The query object is the
    /// original point query object passed into rtcPointQuery, us-
    /// rPtr is an arbitrary pointer to pass input into and store results of the
    /// callback function. The primID, geomID and context (see
    /// rtcInitPointQueryContext for details) can be used to identify the
    /// geometry data of the primitive. 122Embree API Reference
    /// A RTCPointQueryFunction can also be passed directly as an argument to
    /// rtcPointQuery. In this case the callback is invoked for all primitives
    /// in the scene that intersect the query domain. If a callback function
    /// is passed as an argument to rtcPointQuery and (a potentially
    /// different) callback function is set for a ge- ometry with
    /// rtcSetGeometryPointQueryFunction both callback functions are in-
    /// voked and the callback function passed to rtcPointQuery will be called
    /// before the geometry specific callback function.
    /// If instancing is used, the parameter simliarityScale indicates whether
    /// the current instance transform (top element of the stack in context)
    /// is a similarity transformation or not. Similarity transformations
    /// are composed of translation, rotation and uniform scaling and if a
    /// matrix M defines a similarity transformation, there is a scaling
    /// factor D such that for all x,y: dist(Mx, My) = D * dist(x,
    /// y). In this case the parameter scalingFactor is this scaling factor D
    /// and other- wise it is 0. A valid similarity scale (similarityScale >
    /// 0) allows to compute distance information in instance space and
    /// scale the distances into world space (for example, to update the
    /// query radius, see below) by dividing the instance space distance
    /// with the similarity scale. If the current instance transform is not
    /// a similarity transform (similarityScale is 0), the distance computation
    /// has to be performed in world space to ensure correctness. In this
    /// case the instance to world transformations given with the context
    /// should be used to transform the primitive data into world space.
    /// Otherwise, the query location can be trans- formed into instance
    /// space which can be more efficient. If there is no instance
    /// transform, the similarity scale is 1.
    /// The callback function will potentially be called for primitives outside
    /// the query domain for two reasons: First, the callback is invoked for
    /// all primitives inside a BVH leaf node since no geometry data of
    /// primitives is determined internally and therefore individual
    /// primitives are not culled (only their (aggregated) bounding boxes).
    /// Second, in case non similarity transformations are used, the
    /// resulting ellipsoidal query domain (in instance space) is approximated
    /// by its axis aligned bounding box internally and therefore inner
    /// nodes that do not intersect the original domain might intersect the
    /// approximative bounding box which results in unnecessary callbacks.
    /// In any case, the callbacks are conservative, i.e. if a primitive is
    /// inside the query domain a callback will be invoked but the reverse
    /// is not necessarily true.
    /// For efficiency, the radius of the query object can be decreased (in
    /// world space) inside the callback function to improve culling of
    /// geometry during BVH traversal. If the query radius was updated, the
    /// callback function should return true to issue an update of internal
    /// traversal information. Increasing the radius or modifying
    /// the time or position of the query results in undefined behaviour.
    /// Within the callback function, it is safe to call rtcPointQuery again,
    /// for ex- ample when implementing instancing manually. In this case
    /// the instance trans- formation should be pushed onto the stack in
    /// context. Embree will internally compute the point query information
    /// in instance space using the top element of the stack in context when
    /// rtcPointQuery is called. For a reference implementation of a closest
    /// point traversal of triangle meshes using instancing and user defined
    /// instancing see the tutorial *ClosestPoint*.
    pub unsafe fn set_point_query_function(&mut self, query_fn: RTCPointQueryFunction) {
        rtcSetGeometryPointQueryFunction(self.shared.handle, query_fn);
    }

    /// Unsets the point query function for the geometry.
    pub fn unset_point_query_function(&mut self) {
        unsafe {
            rtcSetGeometryPointQueryFunction(self.shared.handle, None);
        }
        // TODO: clear a stored point-query closure in
        // `self.shared.data.callbacks` once `set_point_query_function`
        // accepts a Rust closure instead of a raw fn.
    }

    /// Sets a callback to query the bounding box of user-defined primitives.
    ///
    /// Only a single callback function can be registered per geometry, and
    /// further invocations overwrite the previously set callback function.
    ///
    /// Unregister the callback function by calling
    /// [`GeometryBuilder::unset_bounds_function`].
    ///
    /// The registered bounding box callback function is invoked to calculate
    /// axis- aligned bounding boxes of the primitives of the user-defined
    /// geometry during spatial acceleration structure construction.
    ///
    /// The arguments of the callback closure are:
    ///
    /// - a shared reference to the user data of the geometry
    ///
    /// - the ID of the primitive to calculate the bounds for
    ///
    /// - the time step at which to calculate the bounds
    ///
    /// - a mutable reference to the bounding box where the result should be
    ///   written to
    ///
    /// In a typical usage scenario one binds the user geometry's primitive data
    /// to this callback with
    /// [`set_bounds_function_owned`](Self::set_bounds_function_owned) /
    /// [`set_bounds_function_borrowed`](Self::set_bounds_function_borrowed) (or
    /// captures it in the closure). The callback then receives it as its
    /// `Option<&D>` argument, indexes by `prim_id`, and writes the proper
    /// bounding box for the requested primitive and time to the destination.
    ///
    /// # Thread safety
    ///
    /// Embree may invoke this callback from multiple threads concurrently, for
    /// example during a parallel [`Scene::commit`](crate::Scene::commit),
    /// or when ray queries are issued from several threads on a shared
    /// scene. The closure must therefore be safe to call from several
    /// threads at once and to share across them: it must not depend
    /// on exclusive `&mut` access to its captures, and everything it captures
    /// must be `Send + Sync`. The `Fn + Send + Sync` bounds on the closure
    /// enforce this.
    pub fn set_bounds_function<F, D>(&mut self, bounds: F)
    where
        D: UserData,
        F: Fn(&mut Bounds, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                rtcSetGeometryBoundsFunction(
                    self.shared.handle,
                    trampoline::bounds_function::<F, D>(),
                    ptr::null_mut(),
                );
                self.install_callback(
                    CbKind::UserBounds,
                    ErasedFn::new(bounds),
                    std::ptr::null(),
                    None,
                );
            },
            // Bounds functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// The owned-data variant of
    /// [`set_bounds_function`](Self::set_bounds_function). See
    /// [`set_intersect_filter_function_owned`](Self::set_intersect_filter_function_owned)
    /// for the per-callback owned-vs-borrowed data model. (User geometry only.)
    pub fn set_bounds_function_owned<F, D>(&mut self, bounds: F, data: D)
    where
        D: UserData,
        F: Fn(&mut Bounds, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => {
                let boxed = Box::new(data);
                let ptr = &*boxed as *const D as *const ();
                unsafe {
                    rtcSetGeometryBoundsFunction(
                        self.shared.handle,
                        trampoline::bounds_function::<F, D>(),
                        ptr::null_mut(),
                    );
                }
                self.install_callback(CbKind::UserBounds, ErasedFn::new(bounds), ptr, Some(boxed));
            }
            // Bounds functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// The borrowed-data variant of
    /// [`set_bounds_function`](Self::set_bounds_function). See
    /// [`set_intersect_filter_function_borrowed`](Self::set_intersect_filter_function_borrowed)
    /// for the per-callback owned-vs-borrowed data model. (User geometry only.)
    pub fn set_bounds_function_borrowed<F, D>(&mut self, bounds: F, data: &'buf D)
    where
        D: UserData,
        F: Fn(&mut Bounds, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => {
                let ptr = data as *const D as *const ();
                unsafe {
                    rtcSetGeometryBoundsFunction(
                        self.shared.handle,
                        trampoline::bounds_function::<F, D>(),
                        ptr::null_mut(),
                    );
                }
                self.install_callback(CbKind::UserBounds, ErasedFn::new(bounds), ptr, None);
            }
            // Bounds functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// Unsets the callback to calculate the bounding box of user-defined
    /// geometry.
    pub fn unset_bounds_function(&mut self) {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                rtcSetGeometryBoundsFunction(self.shared.handle, None, ptr::null_mut());
                self.clear_callback(CbKind::UserBounds);
            },
            _ => {}
        }
    }

    /// Sets the callback function to intersect a user geometry.
    ///
    /// The registered
    ///   callback function is invoked by intersect-type ray queries to
    ///   calculate the intersection of a ray packet of variable size with one
    ///   user-defined primitive.
    /// Only a single callback function can be registered per geometry and
    /// further invocations overwrite the previously set callback function.
    /// Unregister the callback function by calling
    /// [`GeometryBuilder::unset_intersect_function`].
    ///
    ///
    /// # Arguments
    ///
    /// - `intersect`: The callback function to register. The task of the
    ///   callback function is to intersect each active ray from the ray packet
    ///   with the specified user primitive. If the user-defined primitive is
    ///   missed by a ray of the ray packet, the function should return without
    ///   modifying the ray or hit. If an intersection of the user-defined
    ///   primitive with the ray is found in the range `tnear` to `tfar`, it
    ///   should update the hit distance of the ray (`tfar` member) and the
    ///   hit(`u`, `v`, `instID`, `geomID`, `primID` members). In particular,
    ///   the currently intersected instance is stored in the `instID` field of
    ///   the intersection context, which must be deep-copied into the `instID`
    ///   member of the hit structure.
    ///
    ///   The callback function gets passed a number of arguments:
    ///     - the ray hit packet of variable size N (see [`RayHitN`]); it
    ///       contains valid data, in particular the `tfar` value is the current
    ///       closest hit distance found. All data inside the `hit` component of
    ///       the ray hit structure are undefined and should **NOT** be read by
    ///       the function.
    ///     - the valid masks for each ray in the packet (see [`ValidityN`])
    ///     - a mutable reference to the intersection context (see
    ///       [`IntersectContext`](`crate::IntersectContext`) and
    ///       [`IntersectContextExt`](`crate::IntersectContextExt`))
    ///     - the geometry ID of the geometry to intersect
    ///     - the primitive ID of the primitive to intersect
    ///     - a shared reference to the user data of the geometry (if any); the
    ///       user data is bound per callback via this setter's `_owned` /
    ///       `_borrowed` variants
    ///
    /// The ray component of the ray hit structure contains valid data, in
    /// particular the tfar value is the current closest hit distance found.
    /// All data inside the hit component of the [`RayHitN`] structure are
    /// undefined and should **NOT** be *read* by the function (writing is ok).
    ///
    /// As a primitive might have multiple intersections with a ray, the
    /// intersection filter function needs to be invoked by the user
    /// geometry intersection callback for each encountered intersection, if
    /// filtering of intersections is desired. This can be achieved through
    /// the [`GeometryBuilder::set_intersect_filter_function`].
    ///
    /// - Within the user geometry intersect function, it is safe to trace new
    ///   rays and create new scenes and geometries.
    ///
    /// - When performing ray queries using [`Scene::intersect`], it is
    ///   guaranteed that the packet size is 1 when the callback is invoked.
    ///
    /// - When performing ray queries using the
    ///   [`Scene::intersect4`]/[`Scene::intersect8`]/[`Scene::intersect16`]
    ///   functions, it is **not** generally guaranteed that the ray packet size
    ///   (and order of rays inside the packet) passed to the callback matches
    ///   the initial ray packet. However, under some circumstances these
    ///   properties are guaranteed, and whether this is the case can be queried
    ///   using [`Device::get_property`].
    ///
    /// - When performing ray queries using the stream API such as
    ///   [`Scene::intersect_stream_soa`], [`Scene::intersect_stream_aos`], the
    ///   order of rays and ray packet size of the callback function might
    ///   change to either 1, 4, 8, or 16.
    ///
    /// - For many usage scenarios, repacking and re-ordering of rays does not
    ///   cause difficulties in implementing the callback function. However,
    ///   algorithms that need to extend the ray with additional data must use
    ///   the rayID component of the ray to identify the original ray to access
    ///   the per-ray data.
    ///
    /// # Thread safety
    ///
    /// Embree may invoke this callback from multiple threads concurrently, for
    /// example during a parallel [`Scene::commit`](crate::Scene::commit),
    /// or when ray queries are issued from several threads on a shared
    /// scene. The closure must therefore be safe to call from several
    /// threads at once and to share across them: it must not depend
    /// on exclusive `&mut` access to its captures, and everything it captures
    /// must be `Send + Sync`. The `Fn + Send + Sync` bounds on the closure
    /// enforce this.
    pub fn set_intersect_function<F, D, C>(&mut self, intersect: F)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut IntersectFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                rtcSetGeometryIntersectFunction(
                    self.shared.handle,
                    trampoline::intersect_function::<F, D, C>(),
                );
                self.install_callback(
                    CbKind::UserIntersect,
                    ErasedFn::new(intersect),
                    std::ptr::null(),
                    None,
                );
            },
            // Intersect functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// The owned-data variant of
    /// [`set_intersect_function`](Self::set_intersect_function). See
    /// [`set_intersect_filter_function_owned`](Self::set_intersect_filter_function_owned)
    /// for the per-callback owned-vs-borrowed data model. (User geometry only.)
    pub fn set_intersect_function_owned<F, D, C>(&mut self, intersect: F, data: D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut IntersectFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                let boxed = Box::new(data);
                let ptr = &*boxed as *const D as *const ();
                rtcSetGeometryIntersectFunction(
                    self.shared.handle,
                    trampoline::intersect_function::<F, D, C>(),
                );
                self.install_callback(
                    CbKind::UserIntersect,
                    ErasedFn::new(intersect),
                    ptr,
                    Some(boxed),
                );
            },
            // Intersect functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// The borrowed-data variant of
    /// [`set_intersect_function`](Self::set_intersect_function). See
    /// [`set_intersect_filter_function_borrowed`](Self::set_intersect_filter_function_borrowed)
    /// for the per-callback owned-vs-borrowed data model. (User geometry only.)
    pub fn set_intersect_function_borrowed<F, D, C>(&mut self, intersect: F, data: &'buf D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut IntersectFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                let ptr = data as *const D as *const ();
                rtcSetGeometryIntersectFunction(
                    self.shared.handle,
                    trampoline::intersect_function::<F, D, C>(),
                );
                self.install_callback(CbKind::UserIntersect, ErasedFn::new(intersect), ptr, None);
            },
            // Intersect functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// Unsets the callback to intersect user-defined geometry.
    pub fn unset_intersect_function(&mut self) {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                rtcSetGeometryIntersectFunction(self.shared.handle, None);
                self.clear_callback(CbKind::UserIntersect);
            },
            // Intersect functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// Sets the callback function to occlude a user geometry.
    ///
    /// Similar to [`GeometryBuilder::set_intersect_function`], but for
    /// occlusion queries.
    ///
    /// # Arguments
    ///
    /// - `occluded`: The callback function to register, which is invoked by
    ///   occlusion queries to test whether the rays of a packet of variable
    ///   size are occluded by a user-defined primitive.  The callback function
    ///   gets passed a number of arguments:
    ///
    ///   - the ray packet of variable size N (see [`RayN`])
    ///   - the valid masks for each ray in the packet (see [`ValidityN`])
    ///   - a mutable reference to the intersection context (see
    ///     [`IntersectContext`](`crate::IntersectContext`) and
    ///     [`IntersectContextExt`](`crate::IntersectContextExt`))
    ///   - the geometry ID of the geometry to intersect
    ///   - the primitive ID of the primitive to intersect
    ///   - a shared reference to the user data of the geometry (if any); the
    ///     user data is bound per callback via this setter's `_owned` /
    ///     `_borrowed` variants
    ///
    /// # Thread safety
    ///
    /// Embree may invoke this callback from multiple threads concurrently, for
    /// example during a parallel [`Scene::commit`](crate::Scene::commit),
    /// or when ray queries are issued from several threads on a shared
    /// scene. The closure must therefore be safe to call from several
    /// threads at once and to share across them: it must not depend
    /// on exclusive `&mut` access to its captures, and everything it captures
    /// must be `Send + Sync`. The `Fn + Send + Sync` bounds on the closure
    /// enforce this.
    pub fn set_occluded_function<F, D, C>(&mut self, occluded: F)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut OccludedFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => {
                unsafe {
                    rtcSetGeometryOccludedFunction(
                        self.shared.handle,
                        trampoline::occluded_function::<F, D, C>(),
                    )
                };
                self.install_callback(
                    CbKind::UserOccluded,
                    ErasedFn::new(occluded),
                    std::ptr::null(),
                    None,
                );
            }
            // Occluded functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// The owned-data variant of
    /// [`set_occluded_function`](Self::set_occluded_function). See
    /// [`set_intersect_filter_function_owned`](Self::set_intersect_filter_function_owned)
    /// for the per-callback owned-vs-borrowed data model. (User geometry only.)
    pub fn set_occluded_function_owned<F, D, C>(&mut self, occluded: F, data: D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut OccludedFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => {
                let boxed = Box::new(data);
                let ptr = &*boxed as *const D as *const ();
                unsafe {
                    rtcSetGeometryOccludedFunction(
                        self.shared.handle,
                        trampoline::occluded_function::<F, D, C>(),
                    )
                };
                self.install_callback(
                    CbKind::UserOccluded,
                    ErasedFn::new(occluded),
                    ptr,
                    Some(boxed),
                );
            }
            // Occluded functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// The borrowed-data variant of
    /// [`set_occluded_function`](Self::set_occluded_function). See
    /// [`set_intersect_filter_function_borrowed`](Self::set_intersect_filter_function_borrowed)
    /// for the per-callback owned-vs-borrowed data model. (User geometry only.)
    ///
    /// # Arguments
    ///
    /// - `occluded`: The callback function to register, which is invoked by
    ///   occlusion queries to test whether the rays of a packet of variable
    ///   size are occluded by a user-defined primitive.
    /// - `data`: A shared reference to the user data of the geometry, which is
    ///  passed to the callback when invoked. The caller must ensure that the
    /// geometry does not outlive the data.
    pub fn set_occluded_function_borrowed<F, D, C>(&mut self, occluded: F, data: &'buf D)
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut OccludedFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::USER => {
                let ptr = data as *const D as *const ();
                unsafe {
                    rtcSetGeometryOccludedFunction(
                        self.shared.handle,
                        trampoline::occluded_function::<F, D, C>(),
                    )
                };
                self.install_callback(CbKind::UserOccluded, ErasedFn::new(occluded), ptr, None);
            }
            // Occluded functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// Unsets the callback to occlude user-defined geometry.
    pub fn unset_occluded_function(&mut self) {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                rtcSetGeometryOccludedFunction(self.shared.handle, None);
                self.clear_callback(CbKind::UserOccluded);
            },
            // Occluded functions apply only to user geometry; ignored otherwise.
            _ => {}
        }
    }

    /// Sets the number of primitives of a user-defined geometry.
    pub fn set_primitive_count(&mut self, count: u32) {
        match self.shared.kind {
            GeometryKind::USER => unsafe {
                rtcSetGeometryUserPrimitiveCount(self.shared.handle, count);
            },
            _ => panic!("Only user geometries can have a primitive count!"),
        }
    }

    /// Set the subdivision mode for the topology of the specified subdivision
    /// geometry.
    ///
    /// The subdivision modes can be used to force linear interpolation for
    /// certain parts of the subdivision mesh:
    ///
    /// * [`RTCSubdivisionMode::NO_BOUNDARY`]: Boundary patches are ignored.
    /// This way each rendered patch has a full set of control vertices.
    ///
    /// * [`RTCSubdivisionMode::SMOOTH_BOUNDARY`]: The sequence of boundary
    /// control points are used to generate a smooth B-spline boundary curve
    /// (default mode).
    ///
    /// * [`RTCSubdivisionMode::PIN_CORNERS`]: Corner vertices are pinned to
    /// their location during subdivision.
    ///
    /// * [`RTCSubdivisionMode::PIN_BOUNDARY`]: All vertices at the border are
    /// pinned to their location during subdivision. This way the boundary is
    /// interpolated linearly. This mode is typically used for texturing to also
    /// map texels at the border of the texture to the mesh.
    ///
    /// * [`RTCSubdivisionMode::PIN_ALL`]: All vertices at the border are pinned
    /// to their location during subdivision. This way all patches are linearly
    /// interpolated.
    pub fn set_subdivision_mode(&mut self, topology_id: u32, mode: SubdivisionMode) {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcSetGeometrySubdivisionMode(self.shared.handle, topology_id, mode)
            },
            _ => panic!("Only subdivision geometries can have a subdivision mode!"),
        }
    }

    /// Sets the number of topologies of a subdivision geometry.
    ///
    /// The number of topologies of a subdivision geometry must be greater
    /// or equal to 1.
    ///
    /// To use multiple topologies, first the number of topologies must be
    /// specified, then the individual topologies can be configured using
    /// [`GeometryBuilder::set_subdivision_mode`] and by setting an index buffer
    /// ([`BufferUsage::INDEX`]) using the topology ID as the buffer slot.
    pub fn set_topology_count(&mut self, count: u32) {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcSetGeometryTopologyCount(self.shared.handle, count);
            },
            _ => panic!("Only subdivision geometries can have multiple topologies!"),
        }
    }

    /// The geometry kind. Mirrors [`Geometry::kind`] for the build phase.
    pub fn kind(&self) -> GeometryKind { self.shared.kind }

    /// The raw Embree geometry handle. Mirrors [`Geometry::handle`].
    ///
    /// # Safety
    ///
    /// The handle is not reference-counted by this call and must not outlive
    /// the geometry.
    pub unsafe fn handle(&self) -> RTCGeometry { self.shared.handle }

    /// The buffer bound to the given slot/usage. Mirrors
    /// [`Geometry::get_buffer`].
    pub fn get_buffer(&self, usage: BufferUsage, slot: u32) -> Option<BufferSource<'_>> {
        self.shared.buffer_source(usage, slot)
    }

    /// Maps a geometry-local buffer slot for **exclusive writing** (re-fill a
    /// buffer created with
    /// [`set_new_buffer`](GeometryBuilder::set_new_buffer)). Sound because
    /// the builder is the unique owner: `&mut self` is genuine exclusive
    /// access. `Err(INVALID_ARGUMENT)` if the slot is unbound / not a local
    /// buffer, or the `T` layout checks fail.
    pub fn map_buffer_mut<T: BufferData>(
        &mut self,
        usage: BufferUsage,
        slot: u32,
    ) -> Result<BufferViewMut<'_, T>, Error> {
        let (ptr, len) = self.shared.map_local::<T>(usage, slot)?;
        // SAFETY: `map_local` validated layout/alignment; `&mut self` (unique builder)
        // gives exclusive access for the view's borrow.
        Ok(unsafe { BufferViewMut::from_raw_parts(ptr, len) })
    }

    /// Maps a geometry-local buffer slot for reading.
    pub fn map_buffer<T: BufferData>(
        &self,
        usage: BufferUsage,
        slot: u32,
    ) -> Result<BufferView<'_, T>, Error> {
        let (ptr, len) = self.shared.map_local::<T>(usage, slot)?;
        // SAFETY: validated; shared borrow of `self` for the view.
        Ok(unsafe { BufferView::from_raw_parts(ptr, len) })
    }

    /// Sets the number of primitives of a user-defined geometry.
    pub fn set_user_primitive_count(&mut self, count: u32) {
        match self.shared.kind {
            GeometryKind::USER => {
                // Update the primitive count.
                unsafe {
                    rtcSetGeometryUserPrimitiveCount(self.shared.handle, count);
                }
            }
            // Primitive count is meaningful only for user-defined geometry; a
            // no-op for every other kind.
            _ => {}
        }
    }

    /// Binds a vertex attribute to a topology of the geometry.
    ///
    /// This function binds a vertex attribute buffer slot to a topology for the
    /// specified subdivision geometry. Standard vertex buffers are always bound
    /// to the default topology (topology 0) and cannot be bound
    /// differently. A vertex attribute buffer always uses the topology it
    /// is bound to when used in the `rtcInterpolate` and `rtcInterpolateN`
    /// calls.
    ///
    /// A topology with ID `i` consists of a subdivision mode set through
    /// `GeometryBuilder::set_subdivision_mode` and the index buffer bound to
    /// the index buffer slot `i`. This index buffer can assign indices for
    /// each face of the subdivision geometry that are different to the
    /// indices of the default topology. These new indices can for example
    /// be used to introduce additional borders into the subdivision mesh to
    /// map multiple textures onto one subdivision geometry.
    pub fn set_vertex_attribute_topology(&mut self, vertex_attribute_id: u32, topology_id: u32) {
        unsafe {
            rtcSetGeometryVertexAttributeTopology(
                self.shared.handle,
                vertex_attribute_id,
                topology_id,
            );
        }
    }

    /// Sets the displacement function for a subdivision geometry.
    ///
    /// Only one displacement function can be set per geometry, further calls to
    /// this will overwrite the previous displacement function. Use
    /// [`GeometryBuilder::unset_displacement_function`] to remove the
    /// displacement function.
    ///
    /// The registered function is invoked to displace points on the subdivision
    /// geometry during spatial acceleration structure construction,
    /// during the [`Scene::commit`] call.
    ///
    /// # Arguments
    ///
    /// * `displacement`: The displacement function. The displacement function
    ///   is called for each vertex of the subdivision geometry.
    ///
    ///   The function is called with the following parameters:
    ///
    ///   * `geometry`: The raw geometry handle [`sys::RTCGeometry`].
    ///   * `vertices`: The information about the vertices to displace. See
    ///     [`Vertices`].
    ///   * `prim_id`: The ID of the primitive that contains the vertices to
    ///     displace.
    ///   * `time_step`: The time step for which the displacement function is
    ///     evaluated. Important for time dependent displacement and motion
    ///     blur.
    ///   * `user_data`: the data bound to this callback via its `_owned` /
    ///     `_borrowed` variant, or `None`.
    ///
    /// # Safety
    ///
    /// The callback function provided to this function contains a raw pointer
    /// to Embree geometry.
    ///
    /// # Thread safety
    ///
    /// Embree may invoke this callback from multiple threads concurrently
    /// during a parallel [`Scene::commit`](crate::Scene::commit). The
    /// closure must therefore be safe to call from several threads at once
    /// and to share across them: it must not depend on exclusive `&mut`
    /// access to its captures, and everything it captures must be `Send +
    /// Sync`. The `Fn + Send + Sync` bounds on the closure enforce this.
    pub unsafe fn set_displacement_function<F, D>(&mut self, displacement: F)
    where
        D: UserData,
        F: for<'a> Fn(RTCGeometry, Vertices<'a>, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => {
                unsafe {
                    rtcSetGeometryDisplacementFunction(
                        self.shared.handle,
                        trampoline::displacement_function::<F, D>(),
                    )
                }
                self.install_callback(
                    CbKind::Displacement,
                    ErasedFn::new(displacement),
                    std::ptr::null(),
                    None,
                );
            }
            // Displacement functions apply only to subdivision geometry; ignored
            // otherwise.
            _ => {}
        }
    }

    /// The owned-data variant of
    /// [`set_displacement_function`](Self::set_displacement_function). See
    /// [`set_intersect_filter_function_owned`](Self::set_intersect_filter_function_owned)
    /// for the per-callback owned-vs-borrowed data model. (Subdivision only.)
    ///
    /// # Safety
    ///
    /// Same contract as
    /// [`set_displacement_function`](Self::set_displacement_function).
    pub unsafe fn set_displacement_function_owned<F, D>(&mut self, displacement: F, data: D)
    where
        D: UserData,
        F: for<'a> Fn(RTCGeometry, Vertices<'a>, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => {
                let boxed = Box::new(data);
                let ptr = &*boxed as *const D as *const ();
                unsafe {
                    rtcSetGeometryDisplacementFunction(
                        self.shared.handle,
                        trampoline::displacement_function::<F, D>(),
                    )
                }
                self.install_callback(
                    CbKind::Displacement,
                    ErasedFn::new(displacement),
                    ptr,
                    Some(boxed),
                );
            }
            // Displacement functions apply only to subdivision geometry; ignored
            // otherwise.
            _ => {}
        }
    }

    /// The borrowed-data variant of
    /// [`set_displacement_function`](Self::set_displacement_function). See
    /// [`set_intersect_filter_function_borrowed`](Self::set_intersect_filter_function_borrowed)
    /// for the per-callback owned-vs-borrowed data model. (Subdivision only.)
    ///
    /// # Safety
    ///
    /// Same contract as
    /// [`set_displacement_function`](Self::set_displacement_function).
    pub unsafe fn set_displacement_function_borrowed<F, D>(
        &mut self,
        displacement: F,
        data: &'buf D,
    ) where
        D: UserData,
        F: for<'a> Fn(RTCGeometry, Vertices<'a>, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => {
                let ptr = data as *const D as *const ();
                unsafe {
                    rtcSetGeometryDisplacementFunction(
                        self.shared.handle,
                        trampoline::displacement_function::<F, D>(),
                    )
                }
                self.install_callback(CbKind::Displacement, ErasedFn::new(displacement), ptr, None);
            }
            // Displacement functions apply only to subdivision geometry; ignored
            // otherwise.
            _ => {}
        }
    }

    /// Removes the displacement function for a subdivision geometry.
    pub fn unset_displacement_function(&mut self) {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcSetGeometryDisplacementFunction(self.shared.handle, None);
                self.clear_callback(CbKind::Displacement);
            },
            _ => panic!("Only subdivision geometries can have displacement functions!"),
        }
    }

    /// Sets the instanced scene of an instance geometry.
    pub fn set_instanced_scene(&mut self, scene: &Scene) {
        match self.shared.kind {
            GeometryKind::INSTANCE => unsafe {
                rtcSetGeometryInstancedScene(self.shared.handle, scene.handle)
            },
            _ => panic!("Only instance geometries can have instanced scenes!"),
        }
    }

    /// Sets the transformation for a particular time step of an instance
    /// geometry.
    ///
    /// The transformation is specified as a 4x4 column-major matrix.
    pub fn set_transform(&mut self, time_step: u32, transform: &[f32; 16]) {
        match self.shared.kind {
            GeometryKind::INSTANCE => unsafe {
                rtcSetGeometryTransform(
                    self.shared.handle,
                    time_step,
                    Format::FLOAT4X4_COLUMN_MAJOR,
                    transform.as_ptr() as *const _,
                );
            },
            _ => panic!("Only instance geometries can have instanced scenes!"),
        }
    }

    /// Sets the transformation for a particular time step of an instance
    /// geometry as a decomposition of the transformation matrix using
    /// quaternions to represent the rotation.
    pub fn set_transform_quaternion(
        &mut self,
        time_step: u32,
        transform: &QuaternionDecomposition,
    ) {
        match self.shared.kind {
            GeometryKind::INSTANCE => unsafe {
                rtcSetGeometryTransformQuaternion(
                    self.shared.handle,
                    time_step,
                    transform as &QuaternionDecomposition as *const _,
                );
            },
            _ => panic!("Only instance geometries can have instanced scenes!"),
        }
    }

    /// Checks if the vertex attribute is allowed for the geometry.
    ///
    /// This function do not check if the slot of the vertex attribute.
    fn check_vertex_attribute(&self) -> Result<(), Error> {
        match self.shared.kind {
            GeometryKind::GRID | GeometryKind::USER | GeometryKind::INSTANCE => {
                Err(Error::INVALID_OPERATION)
            }
            _ => Ok(()),
        }
    }

    /// Commits pending changes (`rtcCommitGeometry`) and transitions to the
    /// committed, shareable [`Geometry`] phase.
    ///
    /// Consumes the builder: the geometry can no longer be mutated unless you
    /// later regain a builder via [`Geometry::try_edit`]. The returned
    /// [`Geometry`] can be attached to scenes
    /// ([`Scene::attach_geometry`](crate::Scene::attach_geometry))
    /// and shared across threads for concurrent ray queries.
    pub fn commit(self) -> Geometry<'buf> {
        unsafe {
            rtcCommitGeometry(self.shared.handle);
        }
        Geometry {
            shared: self.shared,
        }
    }

    /// The *owned* data bound to `kind` (set via `set_*_owned`), if its type is
    /// `D`. Borrowed data is not returned here, the application already
    /// owns it. Type check is a free `Any` downcast (no stored `TypeId`).
    pub fn callback_data<D: UserData>(&self, kind: CbKind) -> Option<&D> {
        // SAFETY: a live clone => not sole owner => no concurrent mutation (invariant).
        unsafe {
            (*self.shared.data.owners.get()).owned_data[kind as usize]
                .as_ref()?
                .downcast_ref::<D>()
        }
    }
    /// `&mut` to owned data, during the build phase (unique builder =>
    /// exclusive).
    pub fn callback_data_mut<D: UserData>(&mut self, kind: CbKind) -> Option<&mut D> {
        unsafe {
            (*self.shared.data.owners.get()).owned_data[kind as usize]
                .as_mut()?
                .downcast_mut::<D>()
        }
    }

    fn install_callback(
        &mut self,
        kind: CbKind,
        erased: ErasedFn,
        user_data: *const (),
        owned_data: Option<Box<dyn Any + Send + Sync>>,
    ) {
        unsafe {
            let (site, owners) = self.shared.data_mut();
            let i = kind as usize;
            site.slots[i].closure = erased.as_ptr() as *const ();
            site.slots[i].user_data = user_data;
            owners.closures[i] = Some(erased);
            owners.owned_data[i] = owned_data;
        }
    }

    fn clear_callback(&mut self, kind: CbKind) {
        unsafe {
            let (site, owners) = self.shared.data_mut();
            let i = kind as usize;
            site.slots[i] = Slot::EMPTY;
            owners.closures[i] = None;
            owners.owned_data[i] = None;
        }
    }
}

/// The **committed, shareable phase** of an Embree geometry.
///
/// Obtained from [`GeometryBuilder::commit`] (see [`GeometryBuilder`] for the
/// build phase). A committed geometry is read-only and `Send + Sync + Clone`,
/// so it can be attached to one or more scenes
/// ([`Scene::attach_geometry`](crate::Scene::attach_geometry) /
/// [`Scene::attach_geometry_by_id`](crate::Scene::attach_geometry_by_id)) and
/// shared across threads for concurrent ray queries which matches embree's rule
/// that ray queries are thread-safe as long as nothing is modifying the
/// geometry.
///
/// Only read-only operations live here ([`interpolate`](Geometry::interpolate),
/// [`get_buffer`](Geometry::get_buffer),
/// [`callback_data`](Geometry::callback_data), the half-edge topology queries,
/// …). Every mutator lives on [`GeometryBuilder`].
///
/// To modify a geometry again, regain a [`GeometryBuilder`] with
/// [`try_edit`](Geometry::try_edit), which succeeds only when you are the
/// **sole owner**, so a geometry attached to any scene cannot be edited until
/// it is detached everywhere. For the common render-loop toggles
/// (enable/disable, mark a buffer dirty) *without* detaching, use
/// [`Scene::enable_geometry`](crate::Scene::enable_geometry) /
/// [`Scene::disable_geometry`](crate::Scene::disable_geometry) /
/// [`Scene::update_geometry_buffer`](crate::Scene::update_geometry_buffer).
///
/// It does not own the host buffers bound to it, but it does own the underlying
/// embree geometry object (released when the last clone drops).
#[derive(Debug, Clone)]
pub struct Geometry<'buf> {
    pub(crate) shared: Arc<GeometryShared<'buf>>,
}

unsafe impl<'buf> Send for Geometry<'buf> {}

// SAFETY: `GeometryData`'s `UnsafeCell`s are mutated only through a `&mut
// GeometryBuilder`, the unique `Arc<GeometryShared>` owner (strong_count == 1,
// !Clone, !Sync). No shared observer (other clones, the scene's retained clone,
// Embree's traversal threads) coexists with that mutation; and the ownership
// move / thread handoff that must separate the last write from any later shared
// read is what makes the writes visible (see the module invariant). So a shared
// `&Geometry` only ever observes frozen state. The boxed `F`/`D` are
// `Send + Sync` (enforced at registration).
unsafe impl<'buf> Sync for Geometry<'buf> {}

impl<'buf> Geometry<'buf> {
    /// Creates a new geometry in its mutable build phase (a
    /// [`GeometryBuilder`]). Configure it (buffers, callbacks, …), then
    /// [`GeometryBuilder::commit`] to a shareable [`Geometry`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use embree3::{Device, Geometry, GeometryKind};
    ///
    /// let device = Device::new().unwrap();
    /// let builder = Geometry::new(&device, GeometryKind::TRIANGLE);
    /// let geometry = builder.commit();
    /// ```
    ///
    /// or use the [`Device::create_geometry`] method:
    ///
    /// ```no_run
    /// use embree3::{Device, GeometryKind};
    ///
    /// let device = Device::new().unwrap();
    /// let builder = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    /// ```
    pub fn new<'dev>(device: &'dev Device, kind: GeometryKind) -> GeometryBuilder<'buf> {
        let handle = unsafe { rtcNewGeometry(device.handle, kind) };
        let shared = Arc::new(GeometryShared {
            device: device.clone(),
            handle,
            kind,
            attachments: Mutex::new(HashMap::new()),
            data: GeometryData::default(),
        });
        unsafe {
            rtcSetGeometryUserData(handle, shared.data.call_site.get() as *mut _);
        }
        GeometryBuilder { shared }
    }

    /// Regains the unique mutable [`GeometryBuilder`] to edit this geometry.
    ///
    /// Succeeds only if this is the **sole owner** (not attached to any scene,
    /// no other clone). Returns `Err(self)` if it is shared.
    ///
    /// To fully edit an attached geometry: detach it from every scene first,
    /// then `try_edit`. For cheap dynamic toggles (visibility /
    /// buffer-dirty) on an attached geometry, prefer the
    /// `Scene::{enable,disable}_geometry` / `update_geometry_buffer`
    /// methods, which need no detach.
    ///
    /// # Soundness contract
    ///
    /// Exclusivity is tracked via the `Arc` strong count, i.e. **wrapper
    /// clones**. This is sound because all safe sharing goes through
    /// wrapper clones (including
    /// [`Scene::attach_geometry`](crate::Scene::attach_geometry), which retains
    /// one). Raw [`Geometry::handle`] escapes are outside this guarantee.
    pub fn try_edit(mut self) -> Result<GeometryBuilder<'buf>, Geometry<'buf>> {
        if Arc::get_mut(&mut self.shared).is_some() {
            Ok(GeometryBuilder {
                shared: self.shared,
            })
        } else {
            Err(self)
        }
    }

    /// Returns the raw Embree geometry handle.
    ///
    /// # Safety
    ///
    /// Use this function only if you know what you are doing. The returned
    /// handle is a raw pointer to an Embree reference-counted object. The
    /// reference count is not increased by this function, so the caller must
    /// ensure that the handle is not used after the geometry object is
    /// destroyed.
    pub unsafe fn handle(&self) -> RTCGeometry { self.shared.handle }

    /// Returns the buffer bound to the given slot and usage.
    pub fn get_buffer(&self, usage: BufferUsage, slot: u32) -> Option<BufferSource<'_>> {
        self.shared.buffer_source(usage, slot)
    }

    /// Maps a geometry-local buffer slot for **reading**. The returned view
    /// borrows `&self`, so it cannot coexist with
    /// [`try_edit`](Geometry::try_edit) (which consumes `self`), i.e. the
    /// buffer cannot be rebound while a view is held. Concurrent read views
    /// are fine. `Err(INVALID_ARGUMENT)` if the slot is unbound /
    /// not a local buffer, or the `T` layout checks fail. To *write* an
    /// attached geometry's buffer, go through
    /// [`Scene::with_geometry_buffer_mut`](crate::Scene::with_geometry_buffer_mut).
    pub fn map_buffer<T: BufferData>(
        &self,
        usage: BufferUsage,
        slot: u32,
    ) -> Result<BufferView<'_, T>, Error> {
        let (ptr, len) = self.shared.map_local::<T>(usage, slot)?;
        // SAFETY: validated; shared borrow of `self` for the view.
        Ok(unsafe { BufferView::from_raw_parts(ptr, len) })
    }

    /// Returns the type of geometry of this geometry.
    pub fn kind(&self) -> GeometryKind { self.shared.kind }

    /// The *owned* data bound to `kind` (set via `set_*_owned`), if its type is
    /// `D`. Borrowed data is not returned here, the application already
    /// owns it. Type check is a free `Any` downcast (no stored `TypeId`).
    pub fn callback_data<D: UserData>(&self, kind: CbKind) -> Option<&D> {
        // SAFETY: a live clone => not sole owner => no concurrent mutation (invariant).
        unsafe {
            (*self.shared.data.owners.get()).owned_data[kind as usize]
                .as_ref()?
                .downcast_ref::<D>()
        }
    }

    /// Smoothly interpolates per-vertex data over the geometry.
    ///
    /// This interpolation is supported for triangle meshes, quad meshes, curve
    /// geometries, and subdivision geometries. Apart from interpolating the
    /// vertex at- tribute itself, it is also possible to get the first and
    /// second order derivatives of that value. This interpolation ignores
    /// displacements of subdivision surfaces and always interpolates the
    /// underlying base surface.
    ///
    /// Interpolated values are written to `args.p`, `args.dp_du`, `args.dp_dv`,
    /// `args.ddp_du_du`, `args.ddp_dv_dv`, and `args.ddp_du_dv`. Set them to
    /// `None` if you do not need to interpolate them.
    ///
    /// All output arrays must be padded to 16 bytes.
    pub fn interpolate(&self, input: InterpolateInput, output: &mut InterpolateOutput) {
        let args = RTCInterpolateArguments {
            geometry: self.shared.handle,
            primID: input.prim_id,
            u: input.u,
            v: input.v,
            bufferType: input.usage,
            bufferSlot: input.slot,
            P: output
                .p_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            dPdu: output
                .dp_du_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            dPdv: output
                .dp_dv_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            ddPdudu: output
                .ddp_du_du_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            ddPdvdv: output
                .ddp_dv_dv_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            ddPdudv: output
                .ddp_du_dv_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            valueCount: output.value_count(),
        };
        unsafe {
            rtcInterpolate(&args as _);
        }
    }

    /// Performs N interpolations of vertex attribute data.
    ///
    /// Similar to [`Geometry::interpolate`], but performs N many interpolations
    /// at once. It additionally gets an array of u/v coordinates
    /// [`InterpolateNInput::u/v`]and a valid mask
    /// [`InterpolateNInput::valid`] that specifies which of these
    /// coordinates are valid. The valid mask points to `n` integers, and a
    /// value of -1 denotes valid and 0 invalid.
    ///
    /// If [`InterpolateNInput::valid`] is `None`, all coordinates are
    /// assumed to be valid.
    ///
    /// The destination arrays are filled in structure of array (SOA) layout.
    /// The value [`InterpolateNInput::n`] must be divisible by 4.
    ///
    /// All changes to that geometry must be properly committed.
    pub fn interpolate_n(&self, input: InterpolateNInput, output: &mut InterpolateOutput) {
        assert_eq!(input.n % 4, 0, "N must be a multiple of 4!");
        let args = RTCInterpolateNArguments {
            geometry: self.shared.handle,
            N: input.n,
            valid: input
                .valid
                .as_ref()
                .map(|v| v.as_ptr() as *const _)
                .unwrap_or(ptr::null()),
            primIDs: input.prim_id.as_ptr(),
            u: input.u.as_ptr(),
            v: input.v.as_ptr(),
            bufferType: input.usage,
            bufferSlot: input.slot,
            P: output
                .p_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            dPdu: output
                .dp_du_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            dPdv: output
                .dp_dv_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            ddPdudu: output
                .ddp_du_du_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            ddPdvdv: output
                .ddp_dv_dv_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            ddPdudv: output
                .ddp_du_dv_mut()
                .map(|p| p.as_mut_ptr())
                .unwrap_or(ptr::null_mut()),
            valueCount: output.value_count(),
        };
        unsafe {
            rtcInterpolateN(&args as _);
        }
    }

    /// Returns the first half edge of a face.
    ///
    /// This function can only be used for subdivision meshes. As all topologies
    /// of a subdivision geometry share the same face buffer the function does
    /// not depend on the topology ID.
    pub fn get_first_half_edge(&self, face_id: u32) -> u32 {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcGetGeometryFirstHalfEdge(self.shared.handle, face_id)
            },
            _ => panic!("Only subdivision geometries can have half edges!"),
        }
    }

    /// Returns the face of some half edge.
    ///
    /// This function can only be used for subdivision meshes. As all topologies
    /// of a subdivision geometry share the same face buffer the function does
    /// not depend on the topology ID.
    pub fn get_face(&self, half_edge_id: u32) -> u32 {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcGetGeometryFace(self.shared.handle, half_edge_id)
            },
            _ => panic!("Only subdivision geometries can have half edges!"),
        }
    }

    /// Returns the next half edge of some half edge.
    ///
    /// This function can only be used for subdivision meshes. As all topologies
    /// of a subdivision geometry share the same face buffer the function does
    /// not depend on the topology ID.
    pub fn get_next_half_edge(&self, half_edge_id: u32) -> u32 {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcGetGeometryNextHalfEdge(self.shared.handle, half_edge_id)
            },
            _ => panic!("Only subdivision geometries can have half edges!"),
        }
    }

    /// Returns the previous half edge of some half edge.
    pub fn get_previous_half_edge(&self, half_edge_id: u32) -> u32 {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcGetGeometryPreviousHalfEdge(self.shared.handle, half_edge_id)
            },
            _ => panic!("Only subdivision geometries can have half edges!"),
        }
    }

    /// Returns the opposite half edge of some half edge.
    pub fn get_opposite_half_edge(&self, topology_id: u32, edge_id: u32) -> u32 {
        match self.shared.kind {
            GeometryKind::SUBDIVISION => unsafe {
                rtcGetGeometryOppositeHalfEdge(self.shared.handle, topology_id, edge_id)
            },
            _ => panic!("Only subdivision geometries can have half edges!"),
        }
    }

    /// Returns the interpolated instance transformation for the specified time
    /// step.
    ///
    /// The transformation is returned as a 4x4 column-major matrix.
    pub fn get_transform(&mut self, time: f32) -> [f32; 16] {
        match self.shared.kind {
            GeometryKind::INSTANCE => unsafe {
                let mut transform = [0.0; 16];
                rtcGetGeometryTransform(
                    self.shared.handle,
                    time,
                    Format::FLOAT4X4_COLUMN_MAJOR,
                    transform.as_mut_ptr() as *mut _,
                );
                transform
            },
            _ => panic!("Only instance geometries can have instanced scenes!"),
        }
    }
}

/// The arguments for the `Geometry::interpolate` function.
pub struct InterpolateInput {
    pub prim_id: u32,
    pub u: f32,
    pub v: f32,
    pub usage: BufferUsage,
    pub slot: u32,
}

/// The arguments for the `Geometry::interpolate_n` function.
pub struct InterpolateNInput<'a> {
    pub valid: Option<Cow<'a, [u32]>>,
    pub prim_id: Cow<'a, [u32]>,
    pub u: Cow<'a, [f32]>,
    pub v: Cow<'a, [f32]>,
    pub usage: BufferUsage,
    pub slot: u32,
    pub n: u32,
}

/// The output of the `Geometry::interpolate` and `Geometry::interpolate_n`
/// functions in structure of array (SOA) layout.
pub struct InterpolateOutput {
    /// The buffer containing the interpolated values.
    buffer: Vec<f32>,
    /// The number of values per attribute.
    count_per_attribute: u32,
    /// The offset of the `p` attribute in the buffer.
    p_offset: Option<u32>,
    /// The offset of the `dp_du` attribute in the buffer.
    dp_du_offset: Option<u32>,
    /// The offset of the `dp_dv` attribute in the buffer.
    dp_dv_offset: Option<u32>,
    /// The offset of the `ddp_du_du` attribute in the buffer.
    ddp_du_du_offset: Option<u32>,
    /// The offset of the `ddp_dv_dv` attribute in the buffer.
    ddp_dv_dv_offset: Option<u32>,
    /// The offset of the `ddp_du_dv` attribute in the buffer.
    ddp_du_dv_offset: Option<u32>,
}

impl InterpolateOutput {
    pub fn new(count: u32, zeroth_order: bool, first_order: bool, second_order: bool) -> Self {
        assert!(
            count > 0,
            "The number of interpolated values must be greater than 0!"
        );
        assert!(
            zeroth_order || first_order || second_order,
            "At least one of the origin value, first order derivative, or second order derivative \
             must be true!"
        );
        let mut offset = 0;
        let p_offset = zeroth_order.then(|| {
            let _offset = offset;
            offset += count;
            _offset
        });
        let dp_du_offset = first_order.then(|| {
            let _offset = offset;
            offset += count;
            _offset
        });
        let dp_dv_offset = first_order.then(|| {
            let _offset = offset;
            offset += count;
            _offset
        });
        let ddp_du_du_offset = second_order.then(|| {
            let _offset = offset;
            offset += count;
            _offset
        });
        let ddp_dv_dv_offset = second_order.then(|| {
            let _offset = offset;
            offset += count;
            _offset
        });
        let ddp_du_dv_offset = second_order.then(|| {
            let _offset = offset;
            offset += count;
            _offset
        });

        Self {
            buffer: vec![0.0; (offset + count) as usize],
            count_per_attribute: count,
            p_offset,
            dp_du_offset,
            dp_dv_offset,
            ddp_du_du_offset,
            ddp_dv_dv_offset,
            ddp_du_dv_offset,
        }
    }

    /// Returns the interpolated `p` attribute.
    pub fn p(&self) -> Option<&[f32]> {
        self.p_offset.map(|offset| {
            &self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the mutable interpolated `p` attribute.
    pub fn p_mut(&mut self) -> Option<&mut [f32]> {
        self.p_offset.map(move |offset| {
            &mut self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the interpolated `dp_du` attribute.
    pub fn dp_du(&self) -> Option<&[f32]> {
        self.dp_du_offset.map(|offset| {
            &self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the mutable interpolated `dp_du` attribute.
    pub fn dp_du_mut(&mut self) -> Option<&mut [f32]> {
        self.dp_du_offset.map(|offset| {
            &mut self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the interpolated `dp_dv` attribute.
    pub fn dp_dv(&self) -> Option<&[f32]> {
        self.dp_dv_offset.map(|offset| {
            &self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the mutable interpolated `dp_dv` attribute.
    pub fn dp_dv_mut(&mut self) -> Option<&mut [f32]> {
        self.dp_dv_offset.map(|offset| {
            &mut self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the interpolated `ddp_du_du` attribute.
    pub fn ddp_du_du(&self) -> Option<&[f32]> {
        self.ddp_du_du_offset.map(|offset| {
            &self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the mutable interpolated `ddp_du_du` attribute.
    pub fn ddp_du_du_mut(&mut self) -> Option<&mut [f32]> {
        self.ddp_du_du_offset.map(|offset| {
            &mut self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the interpolated `ddp_dv_dv` attribute.
    pub fn ddp_dv_dv(&self) -> Option<&[f32]> {
        self.ddp_dv_dv_offset.map(|offset| {
            &self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the mutable interpolated `ddp_dv_dv` attribute.
    pub fn ddp_dv_dv_mut(&mut self) -> Option<&mut [f32]> {
        self.ddp_dv_dv_offset.map(move |offset| {
            &mut self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the interpolated `ddp_du_dv` attribute.
    pub fn ddp_du_dv(&self) -> Option<&[f32]> {
        self.ddp_du_dv_offset.map(|offset| {
            &self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the mutable interpolated `ddp_du_dv` attribute.
    pub fn ddp_du_dv_mut(&mut self) -> Option<&mut [f32]> {
        self.ddp_du_dv_offset.map(move |offset| {
            &mut self.buffer[offset as usize..(offset + self.count_per_attribute) as usize]
        })
    }

    /// Returns the number of values per attribute.
    pub fn value_count(&self) -> u32 { self.count_per_attribute }
}

macro_rules! impl_geometry_type {
    ($name:ident, $kind:path, $(#[$meta:meta])*) => {
        /// Typed [`GeometryBuilder`] for a fixed geometry kind. Build via its
        /// `Deref`/`DerefMut` to [`GeometryBuilder`], then `commit` to a
        /// shareable [`Geometry`].
        #[derive(Debug)]
        pub struct $name<'a>(GeometryBuilder<'a>);

        impl<'a> Deref for $name<'a> {
            type Target = GeometryBuilder<'a>;

            fn deref(&self) -> &Self::Target { &self.0 }
        }

        impl<'a> DerefMut for $name<'a> {
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
        }

        $(#[$meta])*
        impl<'a> $name<'a> {
            pub fn new(device: &Device) -> Result<Self, Error> {
                Ok(Self(Geometry::new(device, $kind)))
            }

            /// Commit pending changes and move to the shareable committed phase.
            pub fn commit(self) -> Geometry<'a> { self.0.commit() }
        }
    };
}

impl_geometry_type!(TriangleMeshBuilder, GeometryKind::TRIANGLE,
    /// A triangle mesh geometry builder.
    ///
    /// The index buffer must contain an array of three 32-bit indices per triangle
    /// ([`Format::UINT3`]), and the number of primitives is inferred from the size
    /// of the index buffer.
    ///
    /// The vertex buffer must contain an array of single precision x, y,
    /// and z floating point coordinates per vertex ([`Format::FLOAT3`]), and the
    /// number of vertices is inferred from the size of the vertex buffer.
    /// The vertex buffer can be at most 16 GB in size.
    ///
    /// The parameterization of a triangle uses the first vertex `p0` as the
    /// base point, the vector `p1 - p0` as the u-direction, and the vector
    /// `p2 - p0` as the v-direction. Thus vertex attributes t0, t1, and t2
    /// can be linearly interpolated over the triangle using the barycentric
    /// coordinates `(u,v)` of the hit point:
    ///
    /// t_uv = (1-u-v) * t0 + u * t1 + v * t2
    ///      = t0 + u * (t1 - t0) + v * (t2 - t0)
    ///
    /// A triangle whose vertices are laid out counter-clockwise has its geometry
    /// normal pointing upwards outside the front face.
    ///
    /// For multi-segment motion blur, the number of time steps must be first
    /// specified using the [`GeometryBuilder::set_time_step_count`] call. Then a vertex
    /// buffer for each time step can be set using different buffer slots, and all
    /// these buffers have to have the same stride and size.
);

impl_geometry_type!(QuadMeshBuilder, GeometryKind::QUAD,
    /// A quad mesh geometry builder.
    ///
    /// The index buffer must contain an array of four 32-bit indices per triangle
    /// ([`Format::UINT4`]), and the number of primitives is inferred from the size
    /// of the index buffer.
    ///
    /// The vertex buffer must contain an array of single precision x, y,
    /// and z floating point coordinates per vertex ([`Format::FLOAT3`]), and the
    /// number of vertices is inferred from the size of the vertex buffer.
    /// The vertex buffer can be at most 16 GB in size.
    ///
    /// A quad is internally handled as a pair of two triangles `v0`, `v1`, `v3`
    /// and `v2`, `v3`, `v1`, with the `u'/v'` coordinates of the second triangle
    /// corrected by `u = 1-u'` and `v = 1-v'` to produce a quad parametrization
    /// where `u` and `v` are in the range 0 to 1. Thus the parametrization of a quad
    /// uses the first vertex `p0` as base point, and the vector `p1 - p0` as
    /// u-direction, and `p3 - p0` as v-direction. Thus vertex attributes t0, t1, t2, t3
    /// can be bilinearly interpolated over the quadrilateral the following way:
    ///
    /// t_uv = (1-v)((1-u) * t0 + u * t1) + v * ((1-u) * t3 + u * t2)
    ///
    /// Mixed triangle/quad meshes are supported by encoding a triangle as a quad,
    /// which can be achieved by replicating the last triangle vertex (v0,v1,v2 ->
    /// v0,v1,v2,v2). This way the second triangle is a line (which can never get
    /// hit), and the parametrization of the first triangle is compatible with the
    /// standard triangle parametrization.
    /// A quad whose vertices are laid out counter-clockwise has its geometry
    /// normal pointing upwards outside the front face.
    ///
    ///    p3 ------- p2
    ///    ^          |
    ///  v |          |
    ///    |          |
    ///    p0 ------> p1
    ///        u
);

impl_geometry_type!(UserGeometryBuilder, GeometryKind::USER,
    /// A user geometry builder.
);

impl_geometry_type!(InstanceGeometryBuilder, GeometryKind::INSTANCE,
    /// An instance geometry builder.
);

/// Arguments handed to a user-geometry **intersect** callback registered via
/// [`GeometryBuilder::set_intersect_function`].
///
/// Besides the ray packet and per-callback user data, it carries the machinery
/// to run a candidate hit through the geometry's intersection filter and the
/// context filter via [`filter_intersection`](Self::filter_intersection), which
/// is the only way a user geometry can honour filter functions, since embree
/// cannot auto-invoke them for user-computed hits.
///
/// Per-ray usage (the filter primitive is always `N = 1`; loop the packet):
/// gather lane `i` with [`ray`](Self::ray), build a fully-initialized [`Hit`],
/// set `ray.tfar` to the candidate distance, call
/// [`filter_intersection`](Self::filter_intersection), and on `true` commit
/// with [`commit_hit`](Self::commit_hit). Skip lanes where `valid_n()[i] == 0`.
pub struct IntersectFunctionNArgs<'a, C: AsIntersectContext, D: UserData> {
    // The original FFI args pointer. Required by `rtcFilterIntersection`, and the
    // source of the ray/hit packet (`(*raw).rayhit`) and `N`. Valid only for the
    // duration of the callback invocation; `*const` makes the struct `!Send`/`!Sync`
    // so it cannot escape to another thread.
    raw: *const RTCIntersectFunctionNArguments,
    valid_n: ValidityN<'a>,
    context: &'a mut C,
    geom_id: u32,
    prim_id: u32,
    user_data: Option<&'a D>,
}

impl<'a, C: AsIntersectContext, D: UserData> IntersectFunctionNArgs<'a, C, D> {
    /// Number of rays in the packet.
    pub fn len(&self) -> usize {
        // SAFETY: `raw` is the live args pointer for this callback invocation.
        unsafe { (*self.raw).N as usize }
    }

    /// Whether the packet is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Per-ray validity mask (`0` = inactive lane, skip it; `-1` = active).
    pub fn valid_n(&self) -> &ValidityN<'a> { &self.valid_n }

    /// Mutable validity mask.
    pub fn valid_n_mut(&mut self) -> &mut ValidityN<'a> { &mut self.valid_n }

    /// The intersection context.
    pub fn context(&self) -> &C { self.context }

    /// The intersection context, mutably.
    pub fn context_mut(&mut self) -> &mut C { self.context }

    /// Geometry ID being intersected.
    pub fn geom_id(&self) -> u32 { self.geom_id }

    /// Primitive ID being intersected.
    pub fn prim_id(&self) -> u32 { self.prim_id }

    /// Per-callback user data (if bound via the `_owned` / `_borrowed` setter).
    pub fn user_data(&self) -> Option<&D> { self.user_data }

    // The ray packet (SoA) of the underlying `RTCRayHitN`. The ray block is at
    // offset 0 of the rayhit buffer.
    #[inline]
    fn rays(&self) -> RayN<'a> {
        RayN {
            ptr: unsafe { (*self.raw).rayhit as *mut RTCRayN },
            len: self.len(),
            marker: PhantomData,
        }
    }

    // The hit packet (SoA). `RTCRayHitN` lays the hit block after the 12-float
    // ray block, i.e. at `rayhit + 12 * N` u32s, the same offset `RayHitN::hit_n`
    // uses. NOTE: this is the rayhit's *own* hit block (for scatter); it is NOT
    // the filter's `hit` argument, which is always a separate `Hit` local.
    #[inline]
    fn hits(&self) -> HitN<'a> {
        let n = self.len();
        HitN {
            ptr: unsafe { ((*self.raw).rayhit as *const u32).add(12 * n) as *mut RTCHitN },
            len: n,
            marker: PhantomData,
        }
    }

    /// Gather lane `i` of the packet into a contiguous single-ray [`Ray`]
    /// (≈ embree's `rtcGetRayHitFromRayHitN`, ray part). `ray.tfar` is the
    /// current closest-hit distance for that lane.
    pub fn ray(&self, i: usize) -> Ray {
        debug_assert!(i < self.len(), "ray index out of bounds");
        let r = self.rays();
        let org = r.org(i);
        let dir = r.dir(i);
        Ray {
            org_x: org[0],
            org_y: org[1],
            org_z: org[2],
            tnear: r.tnear(i),
            dir_x: dir[0],
            dir_y: dir[1],
            dir_z: dir[2],
            time: r.time(i),
            tfar: r.tfar(i),
            mask: r.mask(i),
            id: r.id(i),
            flags: r.flags(i),
        }
    }

    /// Run a candidate single-ray `hit` (paired with its `ray`, whose `tfar` is
    /// the candidate distance) through the geometry's intersection filter
    /// **and** the context filter. Returns `true` if the hit survived (then
    /// commit it with [`commit_hit`](Self::commit_hit)); `false` if
    /// rejected.
    ///
    /// `hit` must be **fully initialized** (`Ng_*`, `u`, `v`, `geomID`,
    /// `primID`, `instID`) which are required by the filter.
    pub fn filter_intersection(&mut self, ray: &mut Ray, hit: &mut Hit) -> bool {
        let mut valid: i32 = -1;
        let fargs = RTCFilterFunctionNArguments {
            valid: &mut valid,
            // SAFETY: `raw` is the live args pointer for this callback invocation.
            // `geometryUserPtr` here is the crate's per-geometry `CallSite` pointer,
            // which the filter trampolines already expect.
            geometryUserPtr: unsafe { (*self.raw).geometryUserPtr },
            context: unsafe { (*self.raw).context },
            ray: ray as *mut Ray as *mut RTCRayN,
            hit: hit as *mut Hit as *mut RTCHitN,
            N: 1,
        };
        // SAFETY: `fargs` supplies separate contiguous N=1 `ray` and `hit` buffers
        // (the filter reads `ray` and `hit` as independent pointers, `hit` is NOT
        // `ray + 12*N`; we always pass the candidate `hit`, never one derived from
        // the rayhit). `raw` is the live args pointer.
        unsafe { rtcFilterIntersection(self.raw, &fargs) };
        valid != 0
    }

    /// Scatter an accepted single `hit` and `ray.tfar` back to lane `i` of the
    /// packet (≈ embree's `rtcCopyHitToHitN`). Single-level instancing only
    /// (`instID[0]`).
    pub fn commit_hit(&mut self, i: usize, ray: &Ray, hit: &Hit) {
        debug_assert!(i < self.len(), "commit index out of bounds");
        let mut rays = self.rays();
        rays.set_tfar(i, ray.tfar);
        let mut hits = self.hits();
        hits.set_normal(i, [hit.Ng_x, hit.Ng_y, hit.Ng_z]);
        hits.set_uv(i, [hit.u, hit.v]);
        hits.set_prim_id(i, hit.primID);
        hits.set_geom_id(i, hit.geomID);
        hits.set_inst_id(i, hit.instID[0]);
    }

    /// Set lane `i`'s packet-ray `tfar` (to a candidate distance before a
    /// packet filter, or to restore it after a rejection).
    pub fn set_tfar(&mut self, i: usize, tfar: f32) {
        debug_assert!(i < self.len(), "tfar index out of bounds");
        let mut rays = self.rays();
        rays.set_tfar(i, tfar);
    }

    /// Filter the whole packet in ONE `rtcFilterIntersection` call (`N =
    /// len()`), using the packet's own ray block as the filter ray input.
    ///
    /// - `hits[i]`: candidate [`Hit`] for lane `i`, fully initialized for every
    ///   lane marked active in `valid`. `hits.len()` must equal `len()`.
    /// - `valid[i]`: `-1` = test this lane, `0` = skip. On return `valid[i] ==
    ///   0` means the lane was skipped or rejected; `-1` means it survived.
    ///   `valid.len()` must equal `len()`.
    ///
    /// Set each active lane's `tfar` via [`set_tfar`](Self::set_tfar)
    /// **before** calling (the filter reads the packet ray). This does NOT
    /// commit and does NOT restore `tfar`: commit survivors (e.g.
    /// `commit_hit(i, &self.ray(i), &hits[i])`) and restore `tfar` on
    /// rejected lanes yourself.
    pub fn filter_intersection_n(&mut self, hits: &mut [Hit], valid: &mut [i32]) {
        let n = self.len();
        debug_assert_eq!(hits.len(), n, "hits length must equal packet width");
        debug_assert_eq!(valid.len(), n, "valid length must equal packet width");

        // Stage an N-wide SoA candidate-hit scratch (separate from the packet's hit
        // block, so a rejected lane never leaves a stale hit there). RTCHitN SoA is
        // 8 fields x N: [Ng_x, Ng_y, Ng_z, u, v, primID, geomID, instID], N <= 16.
        let mut scratch = [0u32; 8 * 16];
        {
            let mut hn = HitN {
                ptr: scratch.as_mut_ptr() as *mut RTCHitN,
                len: n,
                marker: PhantomData,
            };
            for i in 0..n {
                if valid[i] == 0 {
                    continue;
                }
                hn.set_normal(i, [hits[i].Ng_x, hits[i].Ng_y, hits[i].Ng_z]);
                hn.set_uv(i, [hits[i].u, hits[i].v]);
                hn.set_prim_id(i, hits[i].primID);
                hn.set_geom_id(i, hits[i].geomID);
                hn.set_inst_id(i, hits[i].instID[0]);
            }
        }

        let fargs = RTCFilterFunctionNArguments {
            valid: valid.as_mut_ptr(),
            // SAFETY: `raw` is the live args pointer for this callback invocation.
            geometryUserPtr: unsafe { (*self.raw).geometryUserPtr },
            context: unsafe { (*self.raw).context },
            // Packet's own ray block, already N-wide SoA, no copy.
            ray: unsafe { (*self.raw).rayhit as *mut RTCRayN },
            hit: scratch.as_mut_ptr() as *mut RTCHitN,
            N: n as u32,
        };
        // SAFETY: `ray` is the packet's N-wide SoA ray block; `hit` is our N-wide SoA
        // candidate scratch; `valid` has N entries; `raw` is live.
        unsafe { rtcFilterIntersection(self.raw, &fargs) };

        // The filter may modify the candidate hits; transpose the scratch back.
        let hn = HitN {
            ptr: scratch.as_mut_ptr() as *mut RTCHitN,
            len: n,
            marker: PhantomData,
        };
        for i in 0..n {
            if valid[i] == 0 {
                continue;
            }
            let ng = hn.normal(i);
            hits[i].Ng_x = ng[0];
            hits[i].Ng_y = ng[1];
            hits[i].Ng_z = ng[2];
            hits[i].u = hn.u(i);
            hits[i].v = hn.v(i);
            hits[i].primID = hn.prim_id(i);
            hits[i].geomID = hn.geom_id(i);
            hits[i].instID[0] = hn.inst_id(i);
        }
    }
}

/// Arguments handed to a user-geometry **occluded** callback registered via
/// [`GeometryBuilder::set_occluded_function`]. The occlusion analogue of
/// [`IntersectFunctionNArgs`]: there is no hit buffer in the packet, so on
/// survival mark the lane occluded with [`set_occluded`](Self::set_occluded).
pub struct OccludedFunctionNArgs<'a, C: AsIntersectContext, D: UserData> {
    raw: *const RTCOccludedFunctionNArguments,
    valid_n: ValidityN<'a>,
    context: &'a mut C,
    geom_id: u32,
    prim_id: u32,
    user_data: Option<&'a D>,
}

impl<'a, C: AsIntersectContext, D: UserData> OccludedFunctionNArgs<'a, C, D> {
    /// Number of rays in the packet.
    pub fn len(&self) -> usize {
        // SAFETY: `raw` is the live args pointer for this callback invocation.
        unsafe { (*self.raw).N as usize }
    }

    /// Whether the packet is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Per-ray validity mask (`0` = inactive lane, skip it; `-1` = active).
    pub fn valid_n(&self) -> &ValidityN<'a> { &self.valid_n }

    /// Mutable validity mask.
    pub fn valid_n_mut(&mut self) -> &mut ValidityN<'a> { &mut self.valid_n }

    /// The intersection context.
    pub fn context(&self) -> &C { self.context }

    /// The intersection context, mutably.
    pub fn context_mut(&mut self) -> &mut C { self.context }

    /// Geometry ID being tested.
    pub fn geom_id(&self) -> u32 { self.geom_id }

    /// Primitive ID being tested.
    pub fn prim_id(&self) -> u32 { self.prim_id }

    /// Per-callback user data (if bound via the `_owned` / `_borrowed` setter).
    pub fn user_data(&self) -> Option<&D> { self.user_data }

    // The ray packet (SoA). For occlusion the args carry a plain `ray` pointer.
    #[inline]
    fn rays(&self) -> RayN<'a> {
        RayN {
            ptr: unsafe { (*self.raw).ray },
            len: self.len(),
            marker: PhantomData,
        }
    }

    /// Gather lane `i` of the packet into a contiguous single-ray [`Ray`].
    pub fn ray(&self, i: usize) -> Ray {
        debug_assert!(i < self.len(), "ray index out of bounds");
        let r = self.rays();
        let org = r.org(i);
        let dir = r.dir(i);
        Ray {
            org_x: org[0],
            org_y: org[1],
            org_z: org[2],
            tnear: r.tnear(i),
            dir_x: dir[0],
            dir_y: dir[1],
            dir_z: dir[2],
            time: r.time(i),
            tfar: r.tfar(i),
            mask: r.mask(i),
            id: r.id(i),
            flags: r.flags(i),
        }
    }

    /// Run a candidate occluder (`ray` + fully-initialized `hit`) through the
    /// geometry's occlusion filter **and** the context filter. Returns `true`
    /// if it survived (then mark the lane occluded via
    /// [`set_occluded`](Self::set_occluded)); `false` if rejected.
    pub fn filter_occlusion(&mut self, ray: &mut Ray, hit: &mut Hit) -> bool {
        let mut valid: i32 = -1;
        let fargs = RTCFilterFunctionNArguments {
            valid: &mut valid,
            // SAFETY: `raw` is the live args pointer for this callback invocation.
            geometryUserPtr: unsafe { (*self.raw).geometryUserPtr },
            context: unsafe { (*self.raw).context },
            ray: ray as *mut Ray as *mut RTCRayN,
            hit: hit as *mut Hit as *mut RTCHitN,
            N: 1,
        };
        // SAFETY: separate contiguous N=1 ray/hit buffers; `raw` is live (see
        // `filter_intersection`).
        unsafe { rtcFilterOcclusion(self.raw, &fargs) };
        valid != 0
    }

    /// Mark lane `i` occluded by setting its `tfar` to `-inf` (embree's
    /// occlusion convention).
    pub fn set_occluded(&mut self, i: usize) {
        debug_assert!(i < self.len(), "occluded index out of bounds");
        let mut rays = self.rays();
        rays.set_tfar(i, f32::NEG_INFINITY);
    }

    /// Set lane `i`'s packet-ray `tfar` (candidate distance before a packet
    /// filter, or to restore it after a rejection).
    pub fn set_tfar(&mut self, i: usize, tfar: f32) {
        debug_assert!(i < self.len(), "tfar index out of bounds");
        let mut rays = self.rays();
        rays.set_tfar(i, tfar);
    }

    /// Packet occlusion filter: ONE `rtcFilterOcclusion` call (`N = len()`)
    /// using the packet's own ray block. Same `hits` / `valid` contract as
    /// [`IntersectFunctionNArgs::filter_intersection_n`]. Set active lanes'
    /// `tfar` via [`set_tfar`](Self::set_tfar) first; mark survivors
    /// occluded with [`set_occluded`](Self::set_occluded) and restore
    /// `tfar` on rejected lanes.
    pub fn filter_occlusion_n(&mut self, hits: &mut [Hit], valid: &mut [i32]) {
        let n = self.len();
        debug_assert_eq!(hits.len(), n, "hits length must equal packet width");
        debug_assert_eq!(valid.len(), n, "valid length must equal packet width");

        // N-wide SoA candidate-hit scratch (see filter_intersection_n for layout).
        let mut scratch = [0u32; 8 * 16];
        {
            let mut hn = HitN {
                ptr: scratch.as_mut_ptr() as *mut RTCHitN,
                len: n,
                marker: PhantomData,
            };
            for i in 0..n {
                if valid[i] == 0 {
                    continue;
                }
                hn.set_normal(i, [hits[i].Ng_x, hits[i].Ng_y, hits[i].Ng_z]);
                hn.set_uv(i, [hits[i].u, hits[i].v]);
                hn.set_prim_id(i, hits[i].primID);
                hn.set_geom_id(i, hits[i].geomID);
                hn.set_inst_id(i, hits[i].instID[0]);
            }
        }

        let fargs = RTCFilterFunctionNArguments {
            valid: valid.as_mut_ptr(),
            // SAFETY: `raw` is the live args pointer for this callback invocation.
            geometryUserPtr: unsafe { (*self.raw).geometryUserPtr },
            context: unsafe { (*self.raw).context },
            // Occlusion args carry a plain `ray` pointer (N-wide SoA).
            ray: unsafe { (*self.raw).ray },
            hit: scratch.as_mut_ptr() as *mut RTCHitN,
            N: n as u32,
        };
        // SAFETY: `ray` is the packet's N-wide SoA ray block; `hit` is our N-wide SoA
        // candidate scratch; `valid` has N entries; `raw` is live.
        unsafe { rtcFilterOcclusion(self.raw, &fargs) };

        let hn = HitN {
            ptr: scratch.as_mut_ptr() as *mut RTCHitN,
            len: n,
            marker: PhantomData,
        };
        for i in 0..n {
            if valid[i] == 0 {
                continue;
            }
            let ng = hn.normal(i);
            hits[i].Ng_x = ng[0];
            hits[i].Ng_y = ng[1];
            hits[i].Ng_z = ng[2];
            hits[i].u = hn.u(i);
            hits[i].v = hn.v(i);
            hits[i].primID = hn.prim_id(i);
            hits[i].geomID = hn.geom_id(i);
            hits[i].instID[0] = hn.inst_id(i);
        }
    }
}

mod trampoline {
    use super::*;

    /// Helper function to convert a Rust closure to `RTCFilterFunctionN`
    /// callback for intersect.
    pub(crate) fn intersect_filter_function<F, D, C>() -> RTCFilterFunctionN
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        unsafe extern "C" fn inner<F, D, C>(args: *const RTCFilterFunctionNArguments)
        where
            D: UserData,
            C: AsIntersectContext,
            F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
                + Send
                + Sync
                + 'static,
        {
            let site = &*((*args).geometryUserPtr as *const CallSite);
            let slot = site.slots[CbKind::IntersectFilter as usize];
            if slot.closure.is_null() {
                return;
            }

            // SAFETY: `closure` is the boxed `F` (stable until Drop, which cannot run while
            // a trampoline does). For a ZST `F`, this reads no memory.
            let cb = &*(slot.closure as *const F);
            let user_data = if slot.user_data.is_null() {
                None
            } else {
                // SAFETY: bound at registration with this exact `D` (the setter is generic
                // over the same `D` as this monomorphized trampoline), so the cast is sound
                // by construction, no runtime type check needed.
                Some(&*(slot.user_data as *const D))
            };

            let len = (*args).N as usize;
            cb(
                RayN {
                    ptr: (*args).ray,
                    len,
                    marker: PhantomData,
                },
                HitN {
                    ptr: (*args).hit,
                    len,
                    marker: PhantomData,
                },
                ValidityN {
                    ptr: (*args).valid,
                    len,
                    marker: PhantomData,
                },
                &mut *((*args).context as *mut _ as *mut C),
                user_data,
            );
        }
        Some(inner::<F, D, C>)
    }

    /// Helper function to convert a Rust closure to `RTCFilterFunctionN`
    /// callback for occluded.
    pub(crate) fn occluded_filter_function<F, D, C>() -> RTCFilterFunctionN
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
            + Send
            + Sync
            + 'static,
    {
        unsafe extern "C" fn inner<F, D, C>(args: *const RTCFilterFunctionNArguments)
        where
            D: UserData,
            C: AsIntersectContext,
            F: for<'a> Fn(RayN<'a>, HitN<'a>, ValidityN<'a>, &mut C, Option<&D>)
                + Send
                + Sync
                + 'static,
        {
            let site = &*((*args).geometryUserPtr as *const CallSite);
            let slot = site.slots[CbKind::OccludedFilter as usize];
            if slot.closure.is_null() {
                return;
            }

            // SAFETY: `closure` is the boxed `F` (stable until Drop, which cannot run while
            // a trampoline does). For a ZST `F`, this reads no memory.
            let cb = &*(slot.closure as *const F);
            let user_data = if slot.user_data.is_null() {
                None
            } else {
                // SAFETY: bound at registration with this exact `D` (the setter is generic
                // over the same `D` as this monomorphized trampoline), so the cast is sound
                // by construction, no runtime type check needed.
                Some(&*(slot.user_data as *const D))
            };

            let len = (*args).N as usize;
            cb(
                RayN {
                    ptr: (*args).ray,
                    len,
                    marker: PhantomData,
                },
                HitN {
                    ptr: (*args).hit,
                    len,
                    marker: PhantomData,
                },
                ValidityN {
                    ptr: (*args).valid,
                    len,
                    marker: PhantomData,
                },
                &mut *((*args).context as *mut _ as *mut C),
                user_data,
            );
        }
        Some(inner::<F, D, C>)
    }

    /// Helper function to convert a Rust closure to `RTCIntersectFunctionN`
    /// callback.
    pub(crate) fn intersect_function<F, D, C>() -> RTCIntersectFunctionN
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut IntersectFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        unsafe extern "C" fn inner<F, D, C>(args: *const RTCIntersectFunctionNArguments)
        where
            D: UserData,
            C: AsIntersectContext,
            F: for<'a> Fn(&mut IntersectFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
        {
            let site = &*((*args).geometryUserPtr as *const CallSite);
            let slot = site.slots[CbKind::UserIntersect as usize];
            if slot.closure.is_null() {
                return;
            }

            // SAFETY: `closure` is the boxed `F` (stable until Drop, which cannot run while
            // a trampoline does). For a ZST `F`, this reads no memory.
            let cb = &*(slot.closure as *const F);
            let user_data = if slot.user_data.is_null() {
                None
            } else {
                // SAFETY: bound at registration with this exact `D` (the setter is generic
                // over the same `D` as this monomorphized trampoline), so the cast is sound
                // by construction, no runtime type check needed.
                Some(&*(slot.user_data as *const D))
            };
            let len = (*args).N as usize;

            cb(&mut IntersectFunctionNArgs {
                raw: args,
                valid_n: ValidityN {
                    ptr: (*args).valid,
                    len,
                    marker: PhantomData,
                },
                context: &mut *((*args).context as *mut _ as *mut C),
                geom_id: (*args).geomID,
                prim_id: (*args).primID,
                user_data,
            })
        }

        Some(inner::<F, D, C>)
    }

    /// Helper function to convert a Rust closure to `RTCBoundsFunction`
    /// callback.
    pub(crate) fn bounds_function<F, D>() -> RTCBoundsFunction
    where
        D: UserData,
        F: Fn(&mut Bounds, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        unsafe extern "C" fn inner<F, D>(args: *const RTCBoundsFunctionArguments)
        where
            D: UserData,
            F: Fn(&mut Bounds, u32, u32, Option<&D>) + Send + Sync + 'static,
        {
            let site = &*((*args).geometryUserPtr as *const CallSite);
            let slot = site.slots[CbKind::UserBounds as usize];
            if slot.closure.is_null() {
                return;
            }

            // SAFETY: `closure` is the boxed `F` (stable until Drop, which cannot run while
            // a trampoline does). For a ZST `F`, this reads no memory.
            let cb = &*(slot.closure as *const F);
            let user_data = if slot.user_data.is_null() {
                None
            } else {
                // SAFETY: bound at registration with this exact `D` (the setter is generic
                // over the same `D` as this monomorphized trampoline), so the cast is sound
                // by construction, no runtime type check needed.
                Some(&*(slot.user_data as *const D))
            };

            cb(
                &mut *(*args).bounds_o,
                (*args).primID,
                (*args).timeStep,
                user_data,
            );
        }

        Some(inner::<F, D>)
    }

    /// Helper function to convert a Rust closure to `RTCOccludedFunctionN`
    /// callback.
    pub(crate) fn occluded_function<F, D, C>() -> RTCOccludedFunctionN
    where
        D: UserData,
        C: AsIntersectContext,
        F: for<'a> Fn(&mut OccludedFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
    {
        unsafe extern "C" fn inner<F, D, C>(args: *const RTCOccludedFunctionNArguments)
        where
            D: UserData,
            C: AsIntersectContext,
            F: for<'a> Fn(&mut OccludedFunctionNArgs<'a, C, D>) + Send + Sync + 'static,
        {
            let site = &*((*args).geometryUserPtr as *const CallSite);
            let slot = site.slots[CbKind::UserOccluded as usize];
            if slot.closure.is_null() {
                return;
            }

            // SAFETY: `closure` is the boxed `F` (stable until Drop, which cannot run while
            // a trampoline does). For a ZST `F`, this reads no memory.
            let cb = &*(slot.closure as *const F);
            let user_data = if slot.user_data.is_null() {
                None
            } else {
                // SAFETY: bound at registration with this exact `D` (the setter is generic
                // over the same `D` as this monomorphized trampoline), so the cast is sound
                // by construction, no runtime type check needed.
                Some(&*(slot.user_data as *const D))
            };

            cb(&mut OccludedFunctionNArgs {
                raw: args,
                valid_n: ValidityN {
                    ptr: (*args).valid,
                    len: (*args).N as usize,
                    marker: PhantomData,
                },
                context: &mut *((*args).context as *mut _ as *mut C),
                geom_id: (*args).geomID,
                prim_id: (*args).primID,
                user_data,
            })
        }

        Some(inner::<F, D, C>)
    }

    /// Helper function to convert a Rust closure to `RTCDisplacementFunctionN`
    /// callback.
    pub(crate) fn displacement_function<F, D>() -> RTCDisplacementFunctionN
    where
        D: UserData,
        F: for<'a> Fn(RTCGeometry, Vertices<'a>, u32, u32, Option<&D>) + Send + Sync + 'static,
    {
        unsafe extern "C" fn inner<F, D>(args: *const RTCDisplacementFunctionNArguments)
        where
            D: UserData,
            F: for<'a> Fn(RTCGeometry, Vertices<'a>, u32, u32, Option<&D>) + Send + Sync + 'static,
        {
            let site = &*((*args).geometryUserPtr as *const CallSite);
            let slot = site.slots[CbKind::Displacement as usize];
            if slot.closure.is_null() {
                return;
            }

            // SAFETY: `closure` is the boxed `F` (stable until Drop, which cannot run while
            // a trampoline does). For a ZST `F`, this reads no memory.
            let cb = &*(slot.closure as *const F);
            let user_data = if slot.user_data.is_null() {
                None
            } else {
                // SAFETY: bound at registration with this exact `D` (the setter is generic
                // over the same `D` as this monomorphized trampoline), so the cast is sound
                // by construction, no runtime type check needed.
                Some(&*(slot.user_data as *const D))
            };

            let len = (*args).N as usize;
            let vertices = Vertices {
                len,
                u: (*args).u,
                v: (*args).v,
                ng_x: (*args).Ng_x,
                ng_y: (*args).Ng_y,
                ng_z: (*args).Ng_z,
                p_x: (*args).P_x,
                p_y: (*args).P_y,
                p_z: (*args).P_z,
                marker: PhantomData,
            };
            cb(
                (*args).geometry,
                vertices,
                (*args).primID,
                (*args).timeStep,
                user_data,
            );
        }

        Some(inner::<F, D>)
    }
}

/// Struct holding data for a set of vertices in SoA layout.
/// This is used as a parameter to the callback function set by
/// [`GeometryBuilder::set_displacement_function`].
pub struct Vertices<'a> {
    /// The number of vertices.
    len: usize,
    /// The u coordinates of points to displace.
    u: *const f32,
    /// The v coordinates of points to displace.
    v: *const f32,
    /// The x components of normal of vertices to displace (normalized).
    ng_x: *const f32,
    ///The y component of normal of vertices to displace (normalized).
    ng_y: *const f32,
    /// The z component of normal of vertices to displace (normalized).
    ng_z: *const f32,
    /// The x components of points to displace.
    p_x: *mut f32,
    /// The y components of points to displace.
    p_y: *mut f32,
    /// The z components of points to displace.
    p_z: *mut f32,
    /// To make sure we don't outlive the lifetime of the pointers.
    marker: PhantomData<&'a mut f32>,
}

impl<'a> Vertices<'a> {
    pub fn into_iter_mut(self) -> VerticesIterMut<'a> {
        VerticesIterMut {
            inner: self,
            cur: 0,
        }
    }
}

pub struct VerticesIterMut<'a> {
    inner: Vertices<'a>,
    cur: usize,
}

impl<'a> Iterator for VerticesIterMut<'a> {
    type Item = ([f32; 2], [f32; 3], [&'a mut f32; 3]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur < self.inner.len {
            unsafe {
                let u = *self.inner.u.add(self.cur);
                let v = *self.inner.v.add(self.cur);
                let ng_x = *self.inner.ng_x.add(self.cur);
                let ng_y = *self.inner.ng_y.add(self.cur);
                let ng_z = *self.inner.ng_z.add(self.cur);
                let p_x = self.inner.p_x.add(self.cur);
                let p_y = self.inner.p_y.add(self.cur);
                let p_z = self.inner.p_z.add(self.cur);
                self.cur += 1;
                Some((
                    [u, v],
                    [ng_x, ng_y, ng_z],
                    [&mut *p_x, &mut *p_y, &mut *p_z],
                ))
            }
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for VerticesIterMut<'a> {
    fn len(&self) -> usize { self.inner.len - self.cur }
}

/// Struct holding data for validity masks used in the callback function set by
/// [`GeometryBuilder::set_intersect_filter_function`],
/// [`GeometryBuilder::set_occluded_filter_function`],
/// [`GeometryBuilder::set_intersect_function`] and
/// [`GeometryBuilder::set_occluded_function`].
///
/// - 0 means it is invalid
/// - -1 means the ray/hit is valid
pub struct ValidityN<'a> {
    ptr: *const i32,
    len: usize,
    marker: PhantomData<&'a [i32]>,
}

pub struct ValidityNIter<'a, 'b> {
    inner: &'b ValidityN<'a>,
    cur: usize,
}

impl<'a> ValidityN<'a> {
    pub fn iter<'b>(&'b self) -> ValidityNIter<'a, 'b> {
        ValidityNIter {
            inner: self,
            cur: 0,
        }
    }

    pub fn iter_mut<'b>(&'b mut self) -> ValidityNIterMut<'a, 'b> {
        ValidityNIterMut {
            inner: self,
            cur: 0,
        }
    }

    pub const fn len(&self) -> usize { self.len }

    pub const fn is_empty(&self) -> bool { self.len == 0 }
}

impl<'a> Index<usize> for ValidityN<'a> {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len, "index out of bounds");
        unsafe { &*self.ptr.add(index) }
    }
}

impl<'a> IndexMut<usize> for ValidityN<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *(self.ptr.add(index) as *mut i32) }
    }
}

impl<'a, 'b> Iterator for ValidityNIter<'a, 'b> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur < self.inner.len {
            unsafe {
                let valid = *self.inner.ptr.add(self.cur);
                self.cur += 1;
                Some(valid)
            }
        } else {
            None
        }
    }
}

pub struct ValidityNIterMut<'a, 'b> {
    inner: &'b mut ValidityN<'a>,
    cur: usize,
}

impl<'a, 'b> Iterator for ValidityNIterMut<'a, 'b> {
    type Item = &'a mut i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur < self.inner.len {
            unsafe {
                let valid = self.inner.ptr.add(self.cur);
                self.cur += 1;
                Some(&mut *(valid as *mut i32))
            }
        } else {
            None
        }
    }
}
