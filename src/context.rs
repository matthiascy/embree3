use crate::sys::*;

/// Trait for extended intersection context enabling passing of additional
/// ray-query specific data.
///
/// # Safety
///
/// Structs that implement this trait must guarantee that they are
/// layout-compatible with [`IntersectContext`] (i.e. pointer casts between the
/// two types are valid). The corresponding pattern in C is called poor man's
/// inheritance. See [`IntersectContextExt`] for an example of how to do this
pub unsafe trait AsIntersectContext {
    /// The per-ray extension payload carried alongside the base context: `()`
    /// for a plain [`IntersectContext`], `E` for an [`IntersectContextExt<E>`].
    type Ext;

    /// The base [`IntersectContext`] embree carries through to callbacks.
    fn as_context(&self) -> &IntersectContext;
    /// Mutable [`as_context`](Self::as_context).
    fn as_mut_context(&mut self) -> &mut IntersectContext;

    /// Raw `*const` to the base context, for the FFI query call.
    fn as_context_ptr(&self) -> *const IntersectContext {
        self.as_context() as *const IntersectContext
    }

    /// Raw `*mut` to the base context, for the FFI query call.
    fn as_mut_context_ptr(&mut self) -> *mut IntersectContext {
        self.as_mut_context() as *mut IntersectContext
    }

    /// The extension payload, if any (`None` for a plain context).
    fn as_extended(&self) -> Option<&Self::Ext>;
    /// Mutable [`as_extended`](Self::as_extended).
    fn as_mut_extended(&mut self) -> Option<&mut Self::Ext>;
}

/// Per ray-query intersection context.
///
/// This is used to configure intersection flags, specify a filter callback
/// function, and specify the chain of IDs of the current instance, and to
/// attach arbitrary user data to the query (e.g. per ray data).
///
/// # Context Filter Callback
///
/// A filter function can be specified inside the context. This function is
/// invoked as a second filter stage after the per-geometry intersect
/// [`GeometryBuilder::set_intersect_filter_function`](crate::GeometryBuilder::set_intersect_filter_function) or occluded filter
/// [`GeometryBuilder::set_occluded_filter_function`](crate::GeometryBuilder::set_occluded_filter_function) function is invoked. Only
/// rays that passed the first filter stage are valid in this second filter
/// stage. Having such a per ray-query filter function can be useful to
/// implement modifications of the behavior of the query, such as collecting all
/// hits or accumulating transparencies.
///
/// It is guaranteed that the intersection context passed to a ray query is
/// directly passed to the registered callback function. This means that it's
/// possible to attach arbitrary user data to the context and access it from the
/// callback (see [`IntersectContextExt`]). On the contrary, the ray is not
/// guaranteed to be passed to the callback functions, thus reading additional
/// data from the ray passed to the callback is not possible.
///
/// ## Note
///
/// The support for the context filter function must be enabled for a scene by
/// using the [`SceneFlags::CONTEXT_FILTER_FUNCTION`](`crate::SceneFlags::CONTEXT_FILTER_FUNCTION`) flag.
///
/// In case of instancing this feature has to get enabled also for each
/// instantiated scene.
///
/// # Hints
///
/// Best primary ray performance can be obtained by using the ray stream API
/// and setting the intersect context flag to
/// [`IntersectContextFlags::COHERENT`](`crate::IntersectContextFlags`). For
/// secondary rays, it is typically better to use the
/// [`IntersectContextFlags::INCOHERENT`](`crate::IntersectContextFlags::INCOHERENT`),
/// unless the rays are known to be coherent(e.g. for primary transparency
/// rays).
pub type IntersectContext = RTCIntersectContext;

impl IntersectContext {
    /// Shortcut to create a IntersectContext with coherent flag set.
    pub fn coherent() -> IntersectContext {
        IntersectContext::new(RTCIntersectContextFlags::COHERENT)
    }

    /// Shortcut to create a IntersectContext with incoherent flag set.
    pub fn incoherent() -> IntersectContext {
        IntersectContext::new(RTCIntersectContextFlags::INCOHERENT)
    }

    /// Create a context with the given traversal `flags`. For the common cases
    /// use [`coherent`](Self::coherent) / [`incoherent`](Self::incoherent).
    pub fn new(flags: RTCIntersectContextFlags) -> IntersectContext {
        RTCIntersectContext {
            flags,
            filter: None,
            instID: [u32::MAX; 1],
        }
    }

    /// Recover the per-ray extension `T` from a context inside a callback.
    ///
    /// Callbacks (geometry intersect/occluded/filter functions) receive the
    /// base `&IntersectContext` that embree carried through unchanged from
    /// the query. If the query passed an [`IntersectContextExt<T>`], its
    /// `ext: T` sits directly after the base context, and this returns it.
    ///
    /// # Safety
    ///
    /// The ray query that produced this callback **must** have used an
    /// [`IntersectContextExt<T>`] with the *same* `T`. This reinterprets the
    /// bytes following the base context as `T`; calling it with a different
    /// `T`, or when the query used a plain [`IntersectContext`], is
    /// undefined behavior. embree gives no way to check this at runtime,
    /// which is why it is `unsafe` and localized to the exact callback that
    /// needs the extension, rather than baked into the callback's type.
    pub unsafe fn ext<T>(&self) -> &T {
        &(*(self as *const IntersectContext as *const IntersectContextExt<T>)).ext
    }

    /// Mutable [`ext`](Self::ext).
    ///
    /// # Safety
    ///
    /// Same as [`ext`](Self::ext): the query must have used a matching
    /// [`IntersectContextExt<T>`].
    pub unsafe fn ext_mut<T>(&mut self) -> &mut T {
        &mut (*(self as *mut IntersectContext as *mut IntersectContextExt<T>)).ext
    }
}

unsafe impl AsIntersectContext for IntersectContext {
    type Ext = ();

    fn as_context(&self) -> &IntersectContext { self }

    fn as_mut_context(&mut self) -> &mut IntersectContext { self }

    fn as_extended(&self) -> Option<&Self::Ext> { None }

    fn as_mut_extended(&mut self) -> Option<&mut Self::Ext> { None }
}

/// Extended intersection context with additional ray query specific data.
///
/// As Embree 3 does not support placing additional data at the end of the ray
/// structure, and accessing that data inside user geometry callbacks and filter
/// callback functions, we have to attach the data to the ray query context.
///
/// Pass an `IntersectContextExt<E>` to a query: the query accepts any
/// `C: `[`AsIntersectContext`] and uses its base [`IntersectContext`] for the
/// FFI call. Callbacks receive only the **base**
/// `&mut IntersectContext` (the callback type is not generic over `E`), so they
/// recover the `E` with the `unsafe` [`IntersectContext::ext`] /
/// [`IntersectContext::ext_mut`] (or, in user intersect/occluded callbacks,
/// [`IntersectFunctionNArgs::context_ext`](crate::IntersectFunctionNArgs::context_ext)
/// / `context_ext_mut`). The recovery is `unsafe` because the callback must
/// know the query used a matching `E` -- see [`IntersectContext::ext`].
#[repr(C)]
#[derive(Debug)]
pub struct IntersectContextExt<E>
where
    E: Sized,
{
    /// The base context embree carries through to callbacks.
    pub ctx: IntersectContext,
    /// The per-ray extension payload, recovered inside callbacks via the
    /// `unsafe` [`IntersectContext::ext`] / [`IntersectContext::ext_mut`].
    pub ext: E,
}

impl<E> Clone for IntersectContextExt<E>
where
    E: Sized + Clone,
{
    fn clone(&self) -> Self {
        IntersectContextExt {
            ctx: self.ctx,
            ext: self.ext.clone(),
        }
    }
}

impl<E> Copy for IntersectContextExt<E> where E: Sized + Copy {}

unsafe impl<E> AsIntersectContext for IntersectContextExt<E>
where
    E: Sized,
{
    type Ext = E;

    fn as_context(&self) -> &IntersectContext { &self.ctx }

    fn as_mut_context(&mut self) -> &mut IntersectContext { &mut self.ctx }

    fn as_extended(&self) -> Option<&Self::Ext> { Some(&self.ext) }

    fn as_mut_extended(&mut self) -> Option<&mut Self::Ext> { Some(&mut self.ext) }
}

impl<E> IntersectContextExt<E>
where
    E: Sized,
{
    /// Create an extended context with traversal `flags` and the per-ray
    /// extension payload `extra`.
    pub fn new(flags: RTCIntersectContextFlags, extra: E) -> IntersectContextExt<E> {
        IntersectContextExt {
            ctx: IntersectContext::new(flags),
            ext: extra,
        }
    }

    /// Extended context with the coherent traversal flag and payload `extra`.
    pub fn coherent(extra: E) -> IntersectContextExt<E> {
        IntersectContextExt {
            ctx: IntersectContext::coherent(),
            ext: extra,
        }
    }

    /// Extended context with the incoherent traversal flag and payload `extra`.
    pub fn incoherent(extra: E) -> IntersectContextExt<E> {
        IntersectContextExt {
            ctx: IntersectContext::incoherent(),
            ext: extra,
        }
    }
}

/// A stack which stores the IDs and instance transformations during a BVH
/// traversal for a point query.
///
/// The transformations are assumed to be affine transformations
/// (3×3 matrix plus translation) and therefore the last column is ignored.
pub type PointQueryContext = RTCPointQueryContext;

impl PointQueryContext {
    /// Creates a point-query context with an empty instance stack.
    ///
    /// Equivalent to embree's inline `rtcInitPointQueryContext` (which is a
    /// header helper, not an exported symbol, so it is reproduced here):
    /// `instStackSize = 0` and the instance ID stack reset to
    /// [`INVALID_ID`](crate::INVALID_ID). The transform stacks are left zeroed;
    /// they are only read once instances push onto the stack during traversal.
    pub fn new() -> Self {
        // Size every stack off the binding's instance-level count so this stays
        // correct if `sys.rs` is regenerated against an embree built with
        // multi-level instancing.
        const LEVELS: usize = RTC_MAX_INSTANCE_LEVEL_COUNT as usize;
        Self {
            world2inst: [[0.0; 16]; LEVELS],
            inst2world: [[0.0; 16]; LEVELS],
            instID: [crate::INVALID_ID; LEVELS],
            instStackSize: 0,
        }
    }
}

impl Default for PointQueryContext {
    fn default() -> Self { Self::new() }
}
