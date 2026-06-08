use crate::{
    callback::ErasedFn, AsIntersectContext, Bounds, BufferData, BufferUsage, BuildQuality,
    Collision, Error, Format, GeometryKind, LinearBounds, PointQuery, PointQuery16, PointQuery4,
    PointQuery8, PointQueryContext, Ray, Ray16, Ray8, RayHit, RayHit16, RayHit8, RayHitNp,
    RayHitPacket, RayPacket, SceneFlags, ValidMask,
};
use std::{collections::HashMap, mem};

use crate::{
    device::Device,
    geometry::Geometry,
    ray::{Ray4, RayHit4, RayNp},
    sys::*,
};

/// A scene containing various geometries.
///
/// `Scene` is a **unique owner** of its `RTCScene` handle: it is deliberately
/// **not `Clone`** (the previous `rtcRetainScene`-based `Clone` let several
/// handles alias one native scene, which made `&mut Scene` non-exclusive and
/// opened a use-after-free between a clone replacing the progress callback and
/// another clone's `commit` invoking it). To share a committed scene across
/// threads for concurrent queries, wrap it in `Arc<Scene>` -- `Scene` is
/// `Send + Sync`, and read-only queries take `&self`. Mutation
/// (`set_progress_monitor_function`, per-frame edits) takes `&mut self`, so the
/// borrow checker now forbids replacing state while a query/commit is running.
#[derive(Debug)]
pub struct Scene<'a> {
    pub(crate) handle: RTCScene,
    pub(crate) device: Device,
    // Plain owned slot (not `Arc<Mutex<_>>`): `Scene` is not `Clone`, so this is
    // never shared, and `set`/`unset` take `&mut self`. embree holds the boxed
    // closure's stable address until it is replaced or the scene drops; exclusive
    // `&mut` access means no `commit` can be reading it concurrently.
    progress_monitor_fn: Option<ErasedFn>,
    // Plain `HashMap` (no `Arc`, no `Mutex`): `Scene` is not `Clone`, so this map
    // is never shared between handles, and every mutator (`attach_geometry`/
    // `detach_geometry`/...) takes `&mut self` while every reader
    // (`get_geometry`/...) takes `&self` -- the borrow checker already serializes
    // them, so no lock is needed. (Concurrent readers via `Arc<Scene>` are plain
    // shared `&` map reads, with no writer possible.)
    geometries: HashMap<u32, Geometry<'a>>,
}

impl<'a> Drop for Scene<'a> {
    fn drop(&mut self) {
        unsafe {
            rtcReleaseScene(self.handle);
        }
    }
}

// SAFETY: `Scene` is a handle to a refcounted embree object that embree permits
// to be traversed from multiple threads concurrently after commit. Its interior
// state: the progress closure (`Send + Sync`, see `ErasedFn`) and the attached
// geometries (whose callbacks/user data are now `Send + Sync`), are safe to
// share. Bound host buffers must, per the caller's contract, outlive the scene
// and be safe to read concurrently during traversal. Mutation goes through
// `&mut self`.
unsafe impl<'a> Sync for Scene<'a> {}
unsafe impl<'a> Send for Scene<'a> {}

/// A per-lane validity mask for the packet query methods (`intersect{4,8,16}`,
/// `occluded{4,8,16}`, `point_query{4,8,16}`): each lane is
/// [`ValidMask::Valid`] (`-1`) or [`ValidMask::Invalid`] (`0`). `N` is the
/// packet width.
///
/// It is `#[repr(C, align(16))]` because embree reads the mask with aligned
/// SIMD loads and (in debug builds) rejects a mask that is not 16-byte aligned
/// -- a plain `[i32; N]` is only 4-byte aligned. Using the [`ValidMask`] enum
/// as the element keeps every lane to embree's required `-1`/`0`. Because the
/// alignment is structural, the mask is handed to embree by pointer with **no
/// copy and no runtime check**.
#[repr(C, align(16))]
pub struct ValidMaskN<const N: usize>([ValidMask; N]);

impl<const N: usize> ValidMaskN<N> {
    /// A mask with every lane [`ValidMask::Valid`].
    pub const fn all_active() -> Self { Self([ValidMask::Valid; N]) }

    /// A mask from explicit per-lane values.
    pub const fn new(lanes: [ValidMask; N]) -> Self { Self(lanes) }

    /// The per-lane values, mutably (e.g. to disable a lane).
    pub fn lanes_mut(&mut self) -> &mut [ValidMask; N] { &mut self.0 }

    /// Raw `const int*` for the FFI call ([`ValidMask`] is `#[repr(i32)]`, so
    /// `[ValidMask; N]` is layout-compatible with `[i32; N]`).
    fn as_ptr(&self) -> *const i32 { self.0.as_ptr().cast() }
}

impl<const N: usize> From<[ValidMask; N]> for ValidMaskN<N> {
    fn from(lanes: [ValidMask; N]) -> Self { Self(lanes) }
}

/// Emits a `point_query{4,8,16}` method. Each lane is an independent query with
/// its own accumulator in `per_lane`; the closure is shared but routed per-lane
/// through the `userPtr` array. See [`Scene::point_query`] for the single-query
/// form and the callback contract.
macro_rules! impl_point_query_packet {
    ($(#[$doc:meta])* $name:ident, $query:ty, $n:literal, $ffi:ident) => {
        $(#[$doc])*
        pub fn $name<F, D>(
            &self,
            valid: &ValidMaskN<$n>,
            queries: &mut $query,
            context: &mut PointQueryContext,
            mut query_fn: F,
            per_lane: &mut [D; $n],
        ) -> bool
        where
            F: FnMut(&mut PointQuery, &mut PointQueryContext, &mut D, u32, u32, f32) -> bool,
        {
            // One callback record per lane: the closure pointer is shared, the
            // `data` pointer is the lane's own accumulator. The records, `query_fn`,
            // and `per_lane` all outlive the synchronous query call below.
            let mut cbdata: [PointQueryCallbackData; $n] = std::array::from_fn(|i| {
                PointQueryCallbackData {
                    scene_closure: &mut query_fn as *mut F as *mut _,
                    data: &mut per_lane[i] as *mut D as *mut _,
                }
            });
            let mut user_ptrs: [*mut std::os::raw::c_void; $n] =
                std::array::from_fn(|i| &mut cbdata[i] as *mut PointQueryCallbackData as *mut _);
            unsafe {
                $ffi(
                    valid.as_ptr(),
                    self.handle,
                    queries as *mut _,
                    context as *mut _,
                    point_query_function_n::<F, D>(),
                    user_ptrs.as_mut_ptr(),
                )
            }
        }
    };
}

impl<'a> Scene<'a> {
    /// Creates a new scene with the given device.
    pub(crate) fn new(device: Device) -> Result<Self, Error> {
        let handle = unsafe { rtcNewScene(device.handle) };
        if handle.is_null() {
            Err(device.get_error())
        } else {
            Ok(Scene {
                handle,
                device,
                geometries: Default::default(),
                progress_monitor_fn: None,
            })
        }
    }

    /// Creates a new scene with the given device and flags.
    pub(crate) fn new_with_flags(device: Device, flags: SceneFlags) -> Result<Self, Error> {
        let mut scene = Self::new(device)?;
        scene.set_flags(flags);
        Ok(scene)
    }

    /// Returns the device the scene got created in.
    pub fn device(&self) -> &Device { &self.device }

    /// Attaches a new geometry to the scene.
    ///
    /// A geometry can get attached to multiple scenes. The geometry ID is
    /// unique per scene, and is used to identify the geometry when hitting
    /// by a ray or ray packet during ray queries.
    ///
    /// Takes `&mut self`: attaching is exclusive, so it cannot run while a
    /// query, commit, or another attach/detach is in progress on this
    /// scene.
    ///
    /// The geometry IDs are assigned sequentially, starting at 0, as long as
    /// no geometries are detached from the scene. If geometries are detached
    /// from the scene, the implementation will reuse IDs in an implementation
    /// dependent way.
    pub fn attach_geometry(&mut self, geometry: &Geometry<'a>) -> u32 {
        let id = unsafe { rtcAttachGeometry(self.handle, geometry.shared.handle) };
        // Retain a wrapper clone so the geometry's `Arc` strong-count is >= 2 while
        // attached. That is what makes `Geometry::try_edit` fail (mutation
        // impossible) until the geometry is detached from every scene.
        self.geometries.insert(id, geometry.clone());
        id
    }

    /// Attaches a geometry to the scene using a specified geometry ID.
    ///
    /// A geometry can get attached to multiple scenes. The user-provided
    /// geometry ID must be unused in the scene, otherwise the creation of the
    /// geometry will fail. Further, the user-provided geometry IDs
    /// should be compact, as Embree internally creates a vector which size is
    /// equal to the largest geometry ID used. Creating very large geometry
    /// IDs for small scenes would thus cause a memory consumption and
    /// performance overhead.
    ///
    /// Takes `&mut self`: attaching is exclusive, so it cannot run while a
    /// query, commit, or another attach/detach is in progress on this
    /// scene.
    pub fn attach_geometry_by_id(&mut self, geometry: &Geometry<'a>, id: u32) {
        unsafe { rtcAttachGeometryByID(self.handle, geometry.shared.handle, id) };
        self.geometries.insert(id, geometry.clone());
    }

    /// Detaches the geometry from the scene.
    ///
    /// Takes `&mut self`: detaching is exclusive, so it cannot run while a
    /// query, commit, or another attach/detach is in progress on this
    /// scene.
    pub fn detach_geometry(&mut self, id: u32) {
        unsafe {
            rtcDetachGeometry(self.handle, id);
        }
        self.geometries.remove(&id);
    }

    /// Returns the geometry bound to the specified geometry ID.
    ///
    /// This function is NOT thread-safe, and thus CAN be used during rendering.
    /// However, it is recommended to store the geometry handle inside the
    /// application's geometry representation and look up the geometry
    /// handle from that representation directly.
    ///
    /// For a thread-safe version of this function, see [`Scene::get_geometry`].
    pub fn get_geometry_unchecked(&self, id: u32) -> Option<Geometry<'a>> {
        let raw = unsafe { rtcGetGeometry(self.handle, id) };
        if raw.is_null() {
            None
        } else {
            self.geometries.get(&id).cloned()
        }
    }

    /// Returns the geometry bound to the specified geometry ID.
    ///
    /// This function is thread safe and should **NOT** get used during
    /// rendering. If you need a fast non-thread safe version during
    /// rendering please use the [`Scene::get_geometry_unchecked`] function.
    pub fn get_geometry(&self, id: u32) -> Option<Geometry<'a>> {
        let raw = unsafe { rtcGetGeometryThreadSafe(self.handle, id) };
        if raw.is_null() {
            None
        } else {
            self.geometries.get(&id).cloned()
        }
    }

    /// Enable a geometry attached at `id`. Requires `&mut self`, so no
    /// `intersect` can be running. Call `commit` afterward for the change
    /// to take effect. (Multi-scene caveat: enable/disable is a
    /// geometry-global flag, so this does not exclude a concurrent
    /// `intersect` on a *different* scene holding the same geometry).
    pub fn enable_geometry(&mut self, id: u32) {
        if let Some(g) = self.geometries.get(&id) {
            unsafe {
                rtcEnableGeometry(g.shared.handle);
            }
        }
    }

    /// Disable a geometry attached at `id` (the counterpart of
    /// [`Scene::enable_geometry`]); same `&mut self` / commit / multi-scene
    /// rules.
    pub fn disable_geometry(&mut self, id: u32) {
        if let Some(g) = self.geometries.get(&id) {
            unsafe { rtcDisableGeometry(g.shared.handle) };
        }
    }

    /// Mark a buffer of the geometry attached at `id` as modified, so embree
    /// re-reads it on the next commit. Requires `&mut self` (no live
    /// traversal).
    pub fn update_geometry_buffer(&mut self, id: u32, usage: BufferUsage, slot: u32) {
        if let Some(g) = self.geometries.get(&id) {
            unsafe { rtcUpdateGeometryBuffer(g.shared.handle, usage, slot) };
        }
    }

    /// Set the transform (column-major 4x4) of the instance geometry attached
    /// at `id`, for the given motion-time step. Requires `&mut self` (no
    /// live traversal); call [`Scene::commit_geometry`] (or rebuild the
    /// scene) for it to take effect. The common per-frame
    /// instance-animation path.
    pub fn set_geometry_transform(&mut self, id: u32, time_step: u32, transform: &[f32; 16]) {
        if let Some(g) = self.geometries.get(&id) {
            unsafe {
                rtcSetGeometryTransform(
                    g.shared.handle,
                    time_step,
                    Format::FLOAT4X4_COLUMN_MAJOR,
                    transform.as_ptr() as *const _,
                );
            }
        }
    }

    /// Commit (`rtcCommitGeometry`) the geometry attached at `id` after a
    /// dynamic edit (e.g. [`Scene::set_geometry_transform`] or
    /// [`Scene::update_geometry_buffer`]). Requires `&mut self` (no live
    /// traversal).
    pub fn commit_geometry(&mut self, id: u32) {
        if let Some(g) = self.geometries.get(&id) {
            unsafe { rtcCommitGeometry(g.shared.handle) };
        }
    }

    /// Mutably maps a geometry-local buffer of the attached geometry `id`,
    /// running `f` with the data as `&mut [T]`, then returning `f`'s
    /// result. The scoped closure is how an *attached* (shared, committed)
    /// geometry's buffer is written: `&mut self` makes the borrow checker
    /// prove no concurrent `intersect`/`commit` (traversal) is running, and
    /// the slice cannot escape `f`. Call
    /// [`update_geometry_buffer`](Scene::update_geometry_buffer) +
    /// [`commit_geometry`](Scene::commit_geometry) afterward for the change to
    /// take effect. `Err(INVALID_ARGUMENT)` if the slot is unbound / not a
    /// local buffer, the `T` checks fail, or no geometry is attached at
    /// `id`.
    pub fn with_geometry_buffer_mut<T: BufferData, R>(
        &mut self,
        id: u32,
        usage: BufferUsage,
        slot: u32,
        f: impl FnOnce(&mut [T]) -> R,
    ) -> Result<R, Error> {
        let geom = self
            .geometries
            .get(&id)
            .cloned()
            .ok_or(Error::INVALID_ARGUMENT)?;
        let (ptr, len) = geom.shared.map_local::<T>(usage, slot)?;
        // SAFETY: `&mut self` excludes concurrent traversal of this scene; `map_local`
        // validated layout/alignment; `geom` (a retained clone) keeps the buffer alive
        // for the call; and the slice cannot escape `f`.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        Ok(f(slice))
    }

    /// Returns the raw underlying handle to the scene, e.g. for passing it to
    /// native code or ISPC kernels.
    ///
    /// # Safety
    ///
    /// Use this function only if you know what you are doing. The returned
    /// handle is a raw pointer to an Embree reference-counted object. The
    /// reference count is not increased by this function, so the caller must
    /// ensure that the handle is not used after the scene object is
    /// destroyed.
    pub unsafe fn handle(&self) -> RTCScene { self.handle }

    /// Commits all changes for the specified scene.
    ///
    /// This internally triggers building of a spatial acceleration structure
    /// for the scene using all available worker threads. After the commit,
    /// ray queries can be executed on the scene.
    ///
    /// If scene geometries get modified or attached or detached, the
    /// [`Scene::commit`] call must be invoked before performing any further
    /// ray queries for the scene; otherwise the effect of the ray query is
    /// undefined.
    ///
    /// The modification of a geometry, committing the scene, and
    /// tracing of rays must always happen sequentially, and never at the
    /// same time.
    ///
    /// Any API call that sets a property of the scene or geometries
    /// contained in the scene count as scene modification, e.g. including
    /// setting of intersection filter functions.
    ///
    /// This crate enforces that sequencing through the borrow checker: `commit`
    /// and the scene mutators take `&mut self` while queries take `&self`, so a
    /// scene shared for concurrent queries via `Arc<Scene>` cannot be committed
    /// or mutated at the same time.
    pub fn commit(&mut self) {
        unsafe {
            rtcCommitScene(self.handle);
        }
    }

    /// Commits the scene from multiple threads.
    ///
    /// This function is similar to [`Scene::commit`], but allows multiple
    /// threads to commit the scene at the same time. All threads must
    /// consistently call [`Scene::join_commit`].
    ///
    /// This method allows a flexible way to lazily create hierarchies
    /// during rendering. A thread reaching a not-yet-constructed sub-scene of a
    /// two-level scene can generate the sub-scene geometry and call this method
    /// on that just generated scene. During construction, further threads
    /// reaching the not-yet-built scene can join the build operation by
    /// also invoking this method. A thread that calls `join_commit` after
    /// the build finishes will directly return from the `join_commit` call.
    ///
    /// Multiple scene commit operations on different scenes can be running at
    /// the same time, hence it is possible to commit many small scenes in
    /// parallel, distributing the commits to many threads.
    ///
    /// Unlike [`commit`](Self::commit), this takes `&self`: its whole purpose
    /// is for **several threads to call it concurrently on the same scene**
    /// (share it via `Arc<Scene>`), all joining one build. That
    /// cooperative-concurrency contract is the caller's to uphold, so it is
    /// `unsafe`. For the common single-threaded case use
    /// [`commit`](Self::commit) (which embree also recommends for
    /// performance).
    ///
    /// # Safety
    ///
    /// embree synchronizes the concurrent `join_commit` callers *among
    /// themselves* (that cooperation is the whole point of the call). What it
    /// does **not** synchronize is a join-commit against other uses of the same
    /// scene. So while any thread is inside `join_commit` for this scene, the
    /// caller must guarantee that **no** thread runs a query
    /// (`intersect`/`occluded`/`point_query`/`collide`), an ordinary
    /// [`commit`](Self::commit), or any mutation
    /// (`attach_geometry`/`detach_geometry`/`set_flags`/`set_build_quality`/
    /// `set_progress_monitor_function`/...) on it. Only other `join_commit`
    /// calls on this scene may run concurrently. Violating this is undefined
    /// behavior.
    pub unsafe fn join_commit(&self) { rtcJoinCommitScene(self.handle); }

    /// Set the scene flags. Multiple flags can be enabled using an OR
    /// operation. See [`SceneFlags`] for all possible flags.
    /// On failure an error code is set that can be queried using
    /// [`rtcGetDeviceError`].
    ///
    /// Possible scene flags are:
    /// - NONE: No flags set.
    /// - DYNAMIC: Provides better build performance for dynamic scenes (but
    ///   also higher memory consumption).
    /// - COMPACT: Uses compact acceleration structures and avoids algorithms
    ///   that consume much memory.
    /// - ROBUST: Uses acceleration structures that allow for robust traversal,
    ///   and avoids optimizations that reduce arithmetic accuracy. This mode is
    ///   typically used for avoiding artifacts caused by rays shooting through
    ///   edges of neighboring primitives.
    /// - CONTEXT_FILTER_FUNCTION: Enables support for a filter function inside
    ///   the intersection context for this scene.
    pub fn set_flags(&mut self, flags: SceneFlags) {
        unsafe {
            rtcSetSceneFlags(self.handle, flags);
        }
    }

    /// Query the flags of the scene.
    ///
    /// Useful when setting individual flags, e.g. to just set the robust mode
    /// without changing other flags the following way:
    /// ```no_run
    /// use embree3::{Device, Scene, SceneFlags};
    /// let device = Device::new().unwrap();
    /// let mut scene = device.create_scene().unwrap();
    /// let flags = scene.get_flags();
    /// scene.set_flags(flags | SceneFlags::ROBUST);
    /// ```
    pub fn get_flags(&self) -> SceneFlags { unsafe { rtcGetSceneFlags(self.handle) } }

    /// Traverses the BVH with a point query object.
    ///
    /// Traverses the BVH using the point query object and calls a user defined
    /// callback function for each primitive of the scene that intersects the
    /// query domain.
    ///
    /// The user has to initialize the query location (x, y and z member) and
    /// query radius in the range [0, ∞]. If the scene contains motion blur
    /// geometries, also the query time (time member) must be initialized to
    /// a value in the range [0, 1].
    ///
    /// # Arguments
    ///
    /// * `query` - The point query object.
    ///
    /// * `context` - The point query context object. It contains ID and
    ///   transformation information of the instancing hierarchy if
    ///   (multilevel-)instancing is used. See [`PointQueryContext`].
    ///
    /// * `query_fn` - Invoked for each primitive whose BVH leaf intersects the
    ///   query domain, with `(query, context, primID, geomID,
    ///   similarityScale)`. It captures its own accumulator (there is no
    ///   separate `data` parameter) and returns `true` if it shrank the query
    ///   radius (so embree updates traversal). Point queries run synchronously
    ///   on the calling thread, so the closure carries no
    ///   `Send`/`Sync`/`'static` bound. For per-geometry logic, branch on
    ///   `geomID` inside the closure. A callback is mandatory: there is no
    ///   callback-less form (it would do nothing), and no separate per-geometry
    ///   point-query function (use `geomID` dispatch here instead).
    ///
    /// Returns embree's `rtcPointQuery` result: `true` if any callback reported
    /// a change to the query.
    ///
    /// The query radius can be decreased inside the callback function, which
    /// allows to efficiently cull parts of the scene during BVH traversal.
    /// Increasing the query radius and modifying time or location of the query
    /// will result in undefined behavior.
    ///
    /// The callback function will be called for all primitives in a leaf node
    /// of the BVH even if the primitive is outside the query domain,
    /// since Embree does not gather geometry information of primitives
    /// internally.
    ///
    /// Point queries can be used with (multi-)instancing. However, care has to
    /// be taken when the instance transformation contains anisotropic scaling
    /// or sheering. In these cases distance computations have to be performed
    /// in world space to ensure correctness and the ellipsoidal query domain
    /// (in instance space) will be approximated with its axis aligned
    /// bounding box internally. Therefore, the callback function might be
    /// invoked even for primitives in inner BVH nodes that do not intersect
    /// the query domain.
    ///
    /// The point query structure must be aligned to 16 bytes.
    ///
    /// Currently, all primitive types are supported by the point query API
    /// except of points, curves and subdivision surfaces.
    ///
    /// See the *ClosestPoint* tutorial in embree's examples for a reference
    /// implementation.
    pub fn point_query<F>(
        &self,
        query: &mut PointQuery,
        context: &mut PointQueryContext,
        mut query_fn: F,
    ) -> bool
    where
        F: FnMut(&mut PointQuery, &mut PointQueryContext, u32, u32, f32) -> bool,
    {
        unsafe {
            rtcPointQuery(
                self.handle,
                query as *mut _,
                context as *mut _,
                point_query_function::<F>(),
                &mut query_fn as *mut F as *mut _,
            )
        }
    }

    impl_point_query_packet!(
        /// Runs 4 independent point queries at once (`rtcPointQuery4`).
        ///
        /// Each lane `i` is its own query (`queries.x[i]`, `radius[i]`, ...) with
        /// its own accumulator `per_lane[i]`; `valid[i] == 0` skips a lane. The
        /// closure runs per active lane with that lane's `&mut D`. See
        /// [`point_query`](Self::point_query) for the callback contract.
        point_query4, PointQuery4, 4, rtcPointQuery4
    );

    impl_point_query_packet!(
        /// Runs 8 independent point queries at once (`rtcPointQuery8`). See
        /// [`point_query4`](Self::point_query4).
        point_query8, PointQuery8, 8, rtcPointQuery8
    );

    impl_point_query_packet!(
        /// Runs 16 independent point queries at once (`rtcPointQuery16`). See
        /// [`point_query4`](Self::point_query4).
        point_query16, PointQuery16, 16, rtcPointQuery16
    );

    /// Set the build quality of the scene. See [`BuildQuality`] for all
    /// possible values.
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
            rtcSetSceneBuildQuality(self.handle, quality);
        }
    }

    /// Register a progress monitor callback function.
    ///
    /// Only one progress monitor callback can be registered per scene,
    /// and further invocations overwrite the previously registered callback.
    ///
    /// Unregister with [`Scene::unset_progress_monitor_function`].
    ///
    /// # Arguments
    ///
    /// * `progress` - A callback function that takes a number in range [0.0,
    ///   1.0] indicating the progress of the operation.
    ///
    /// # Warning
    ///
    /// Must be called after the scene has been committed.
    ///
    /// # Thread safety
    ///
    /// Embree may invoke this callback from multiple threads concurrently
    /// during a parallel [`Scene::commit`](crate::Scene::commit). The
    /// closure must therefore be safe to call from several threads at once
    /// and to share across them: it must not depend on exclusive
    /// `&mut` access to its captures, and everything it captures must be `Send
    /// + Sync`. The `Fn + Send + Sync` bounds on the closure enforce this.
    pub fn set_progress_monitor_function<F>(&mut self, progress: F)
    where
        F: Fn(f64) -> bool + Send + Sync + 'static,
    {
        // `&mut self` is exclusive (Scene is not Clone), so no `commit` can be reading
        // the old closure while we replace it. Storing the new `ErasedFn` drops
        // the old one only after embree has been pointed at the new boxed
        // closure's stable address.
        let erased = ErasedFn::new(progress);
        unsafe {
            rtcSetSceneProgressMonitorFunction(
                self.handle,
                progress_monitor_function::<F>(),
                erased.as_ptr(),
            );
        }
        self.progress_monitor_fn = Some(erased);
    }

    /// Unregister the progress monitor callback function.
    pub fn unset_progress_monitor_function(&mut self) {
        unsafe {
            rtcSetSceneProgressMonitorFunction(self.handle, None, ::std::ptr::null_mut());
        }
        self.progress_monitor_fn = None;
    }

    /// Finds the closest hit of a single ray with the scene.
    ///
    /// Analogous to [`rtcIntersect1`].
    ///
    /// The user has to initialize the ray origin, ray direction, ray segment
    /// (`tnear`, `tfar` ray members), and set the ray flags to 0 (`flags` ray
    /// member). If the scene contains motion blur geometries, also the ray
    /// time (`time` ray member) must be initialized to a value in the range
    /// [0, 1]. If ray masks are enabled at compile time, the ray mask
    /// (`mask` ray member) must be initialized as well. The geometry ID
    /// (`geomID` ray hit member) must be initialized to `INVALID_ID`.
    ///
    /// When no intersection is found, the ray/hit data is not updated. When an
    /// intersection is found, the hit distance is written into the `tfar`
    /// member of the ray and all hit data is set, such as unnormalized
    /// geometry normal in object space (`Ng` hit member), local hit
    /// coordinates (`u, v` hit member), instance ID stack (`instID`
    /// hit member), geometry ID (`geomID` hit member), and primitive ID
    /// (`primID` hit member). See [`RayHit`] for more information.
    ///
    /// The intersection context (`ctx` argument) can specify flags to optimize
    /// traversal and a filter callback function to be invoked for every
    /// intersection. Further, the pointer to the intersection context is
    /// propagated to callback functions invoked during traversal and can
    /// thus be used to extend the ray with additional data. See
    /// [`IntersectContext`](crate::IntersectContext) for more information.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    /// * `ray` - The ray [`RayHit`] to intersect with the scene.
    pub fn intersect<C: AsIntersectContext>(&self, ctx: &mut C, ray: &mut RayHit) {
        unsafe {
            rtcIntersect1(
                self.handle,
                ctx.as_mut_context_ptr(),
                ray as *mut _ as *mut _,
            );
        }
    }

    /// Finds the closest hits for a ray packet of size 4 with the scene.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    /// * `ray` - The ray packet of size 4 to intersect with the scene. The ray
    ///   packet must be aligned to 16 bytes.
    /// * `valid` - Per-lane validity mask ([`ValidMaskN`]): each lane is
    ///   [`ValidMask::Valid`] (`-1`) or [`ValidMask::Invalid`] (`0`).
    ///
    /// The ray packet pointer passed to callback functions is not guaranteed to
    /// be identical to the original ray provided. To extend the ray with
    /// additional data to be accessed in callback functions, use the
    /// intersection context.
    ///
    /// Only active rays are processed, and hit data of inactive rays is not
    /// changed.
    pub fn intersect4<C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        ray: &mut RayHit4,
        valid: &ValidMaskN<4>,
    ) {
        unsafe {
            rtcIntersect4(
                valid.as_ptr(),
                self.handle,
                ctx.as_mut_context_ptr(),
                ray as *mut RTCRayHit4,
            );
        }
    }

    /// Finds the closest hits for a ray packet of size 8 with the scene.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    /// * `ray` - The ray packet of size 8 to intersect with the scene. The ray
    ///   packet must be aligned to 32 bytes.
    /// * `valid` - Per-lane validity mask ([`ValidMaskN`]): each lane is
    ///   [`ValidMask::Valid`] (`-1`) or [`ValidMask::Invalid`] (`0`).
    ///
    /// The ray packet pointer passed to callback functions is not guaranteed to
    /// be identical to the original ray provided. To extend the ray with
    /// additional data to be accessed in callback functions, use the
    /// intersection context.
    ///
    /// Only active rays are processed, and hit data of inactive rays is not
    /// changed.
    pub fn intersect8<C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        ray: &mut RayHit8,
        valid: &ValidMaskN<8>,
    ) {
        unsafe {
            rtcIntersect8(
                valid.as_ptr(),
                self.handle,
                ctx.as_mut_context_ptr(),
                ray as *mut RTCRayHit8,
            );
        }
    }

    /// Finds the closest hits for a ray packet of size 16 with the scene.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    /// * `ray` - The ray packet of size 16 to intersect with the scene. The ray
    ///   packet must be aligned to 64 bytes.
    /// * `valid` - Per-lane validity mask ([`ValidMaskN`]): each lane is
    ///   [`ValidMask::Valid`] (`-1`) or [`ValidMask::Invalid`] (`0`).
    ///
    /// The ray packet pointer passed to callback functions is not guaranteed to
    /// be identical to the original ray provided. To extend the ray with
    /// additional data to be accessed in callback functions, use the
    /// intersection context.
    ///
    /// Only active rays are processed, and hit data of inactive rays is not
    /// changed.
    pub fn intersect16<C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        ray: &mut RayHit16,
        valid: &ValidMaskN<16>,
    ) {
        unsafe {
            rtcIntersect16(
                valid.as_ptr(),
                self.handle,
                ctx.as_mut_context_ptr(),
                ray as *mut RTCRayHit16,
            );
        }
    }

    /// Checks for a single ray if whether there is any hit with the scene.
    ///
    /// When no intersection is found, the ray data is not updated. In case
    /// a hit was found, the `tfar` component of the ray is set to `-inf`.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    ///
    /// * `ray` - The ray to intersect with the scene.
    pub fn occluded<C: AsIntersectContext>(&self, ctx: &mut C, ray: &mut Ray) -> bool {
        unsafe {
            rtcOccluded1(self.handle, ctx.as_mut_context_ptr(), ray as *mut RTCRay);
        }
        ray.tfar == f32::NEG_INFINITY
    }

    /// Checks for each active ray of a ray packet of size 4 if whether there is
    /// any hit with the scene.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    /// * `ray` - The ray packet of size 4 to intersect with the scene. The ray
    ///   packet must be aligned to 16 bytes.
    /// * `valid` - Per-lane validity mask ([`ValidMaskN`]): each lane is
    ///   [`ValidMask::Valid`] (`-1`) or [`ValidMask::Invalid`] (`0`).
    ///
    /// The ray packet pointer passed to callback functions is not guaranteed to
    /// be identical to the original ray provided. To extend the ray with
    /// additional data to be accessed in callback functions, use the
    /// intersection context.
    ///
    /// Only active rays are processed, and hit data of inactive rays is not
    /// changed.
    pub fn occluded4<C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        ray: &mut Ray4,
        valid: &ValidMaskN<4>,
    ) {
        unsafe {
            rtcOccluded4(
                valid.as_ptr(),
                self.handle,
                ctx.as_mut_context_ptr(),
                ray as *mut RTCRay4,
            );
        }
    }

    /// Checks for each active ray of a ray packet of size 4 if whether there is
    /// any hit with the scene.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    /// * `ray` - The ray packet of size 8 to intersect with the scene. The ray
    ///   packet must be aligned to 32 bytes.
    /// * `valid` - Per-lane validity mask ([`ValidMaskN`]): each lane is
    ///   [`ValidMask::Valid`] (`-1`) or [`ValidMask::Invalid`] (`0`).
    ///
    /// The ray packet pointer passed to callback functions is not guaranteed to
    /// be identical to the original ray provided. To extend the ray with
    /// additional data to be accessed in callback functions, use the
    /// intersection context.
    ///
    /// Only active rays are processed, and hit data of inactive rays is not
    /// changed.
    pub fn occluded8<C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        ray: &mut Ray8,
        valid: &ValidMaskN<8>,
    ) {
        unsafe {
            rtcOccluded8(
                valid.as_ptr(),
                self.handle,
                ctx.as_mut_context_ptr(),
                ray as *mut RTCRay8,
            );
        }
    }

    /// Checks for each active ray of a ray packet of size 16 if whether there
    /// is any hit with the scene.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    /// * `ray` - The ray packet of size 16 to intersect with the scene. The ray
    ///   packet must be aligned to 64 bytes.
    /// * `valid` - Per-lane validity mask ([`ValidMaskN`]): each lane is
    ///   [`ValidMask::Valid`] (`-1`) or [`ValidMask::Invalid`] (`0`).
    ///
    /// The ray packet pointer passed to callback functions is not guaranteed to
    /// be identical to the original ray provided. To extend the ray with
    /// additional data to be accessed in callback functions, use the
    /// intersection context.
    ///
    /// Only active rays are processed, and hit data of inactive rays is not
    /// changed.
    pub fn occluded16<C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        ray: &mut Ray16,
        valid: &ValidMaskN<16>,
    ) {
        unsafe {
            rtcOccluded16(
                valid.as_ptr(),
                self.handle,
                ctx.as_mut_context_ptr(),
                ray as *mut RTCRay16,
            );
        }
    }

    /// Finds the closest hits for a stream of M ray packets.
    ///
    /// A ray in the stream is inactive if its `tnear` value is larger than its
    /// `tfar` value. The stream can be any size including zero. Each ray
    /// must be aligned to 16 bytes.
    ///
    /// The implementation of the stream ray query functions may re-order rays
    /// arbitrarily and re-pack rays into ray packets of different size. For
    /// this reason, the callback functions may be invoked with an arbitrary
    /// packet size (of size 1, 4, 8, or 16) and different ordering as
    /// specified initially in the ray stream. For this reason, you MUST NOT
    /// rely on the ordering of the rays in the ray stream to be preserved but
    /// instead use the `rayID` component of the ray to identify the original
    /// rya, e.g. to access a per-ray payload.
    ///
    /// Analogous to [`rtcIntersectNM`] and [`rtcIntersect1M`].
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    ///
    /// * `rays` - The ray stream to intersect with the scene.
    pub fn intersect_stream_aos<P: RayHitPacket, C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        rays: &mut Vec<P>,
    ) {
        let m = rays.len();
        unsafe {
            if P::Ray::LEN == 1 {
                rtcIntersect1M(
                    self.handle,
                    ctx.as_mut_context_ptr(),
                    rays.as_mut_ptr() as *mut _,
                    m as u32,
                    mem::size_of::<P>(),
                );
            } else {
                rtcIntersectNM(
                    self.handle,
                    ctx.as_mut_context_ptr(),
                    rays.as_mut_ptr() as *mut _,
                    P::Ray::LEN as u32,
                    m as u32,
                    mem::size_of::<P>(),
                );
            }
        }
    }

    /// Finds the closest hits for a stream of M ray packets.
    ///
    /// A ray in the stream is inactive if its `tnear` value is larger than its
    /// `tfar` value. The stream can be any size including zero. Each ray
    /// must be aligned to 16 bytes.
    ///
    /// The implementation of the stream ray query functions may re-order rays
    /// arbitrarily and re-pack rays into ray packets of different size. For
    /// this reason, the callback functions may be invoked with an arbitrary
    /// packet size (of size 1, 4, 8, or 16) and different ordering as
    /// specified initially in the ray stream. For this reason, you MUST NOT
    /// rely on the ordering of the rays in the ray stream to be preserved but
    /// instead use the `rayID` component of the ray to identify the original
    /// rya, e.g. to access a per-ray payload.
    ///
    /// Analogous to [`rtcOccluded1M`] and [`rtcOccludedNM`].
    ///
    /// # Arguments
    ///
    /// * `ctx` - The intersection context to use for the ray query. It
    ///   specifies flags to optimize traversal and a filter callback function
    ///   to be invoked for every intersection. Further, the pointer to the
    ///   intersection context is propagated to callback functions invoked
    ///   during traversal and can thus be used to extend the ray with
    ///   additional data. See [`IntersectContext`](crate::IntersectContext) for
    ///   more information.
    ///
    /// * `rays` - The ray stream to intersect with the scene.
    pub fn occluded_stream_aos<P: RayPacket, C: AsIntersectContext>(
        &self,
        ctx: &mut C,
        rays: &mut Vec<P>,
    ) {
        let m = rays.len();
        unsafe {
            if P::LEN == 1 {
                rtcOccluded1M(
                    self.handle,
                    ctx.as_mut_context_ptr(),
                    rays.as_mut_ptr() as *mut RTCRay,
                    m as u32,
                    mem::size_of::<P>(),
                );
            } else {
                rtcOccludedNM(
                    self.handle,
                    ctx.as_mut_context_ptr(),
                    rays.as_mut_ptr() as *mut RTCRayN,
                    P::LEN as u32,
                    m as u32,
                    mem::size_of::<P>(),
                );
            }
        }
    }

    /// Finds the closest hit for a SOA ray stream of size `n`.
    ///
    /// The implementation of the stream ray query functions may re-order rays
    /// arbitrarily and re-pack rays into ray packets of different size. For
    /// this reason, callback functions may be invoked with an arbitrary
    /// packet size (of size 1, 4, 8, or 16) and different ordering as
    /// specified initially. For this reason, one may have to use the rayID
    /// component of the ray to identify the original ray, e.g. to access
    /// a per-ray payload.
    ///
    /// A ray in a ray stream is considered inactive if its tnear value is
    /// larger than its tfar value.
    pub fn intersect_stream_soa<C: AsIntersectContext>(&self, ctx: &mut C, rays: &mut RayHitNp) {
        unsafe {
            rtcIntersectNp(
                self.handle,
                ctx.as_mut_context_ptr(),
                &mut rays.as_raw() as *mut _,
                rays.len() as u32,
            );
        }
    }

    /// Finds any hits for a SOA ray stream of size `n`.
    ///
    /// The implementation of the stream ray query functions may re-order rays
    /// arbitrarily and re-pack rays into ray packets of different size. For
    /// this reason, callback functions may be invoked with an arbitrary
    /// packet size (of size 1, 4, 8, or 16) and different ordering as
    /// specified initially. For this reason, one may have to use the rayID
    /// component of the ray to identify the original ray, e.g. to access
    /// a per-ray payload.
    ///
    /// A ray in a ray stream is considered inactive if its tnear value is
    /// larger than its tfar value.
    pub fn occluded_stream_soa<C: AsIntersectContext>(&self, ctx: &mut C, rays: &mut RayNp) {
        unsafe {
            rtcOccludedNp(
                self.handle,
                ctx.as_mut_context_ptr(),
                &mut rays.as_raw_mut() as *mut RTCRayNp,
                rays.len() as u32,
            );
        }
    }

    /// Finds the closest hit for **M scattered single rays**, passed as an
    /// array of pointers (`rtcIntersect1Mp`).
    ///
    /// Unlike [`intersect_stream_aos`](Self::intersect_stream_aos) (contiguous
    /// `1M` / `NM`), the rays need not be laid out contiguously: each
    /// `&mut RayHit` may live anywhere in memory. Results are written back in
    /// place through the references. A ray is inactive if its `tnear > tfar`.
    pub fn intersect1mp<C: AsIntersectContext>(&self, ctx: &mut C, rayhits: &mut [&mut RayHit]) {
        // Build the array of pointers embree expects (`*mut *mut RTCRayHit`). The
        // `&mut [&mut RayHit]` borrow keeps every target valid and exclusive for
        // the call.
        let mut ptrs: Vec<*mut RTCRayHit> = rayhits
            .iter_mut()
            .map(|r| &mut **r as *mut RTCRayHit)
            .collect();
        unsafe {
            rtcIntersect1Mp(
                self.handle,
                ctx.as_mut_context_ptr(),
                ptrs.as_mut_ptr(),
                ptrs.len() as u32,
            );
        }
    }

    /// Tests occlusion for **M scattered single rays**, passed as an array of
    /// pointers (`rtcOccluded1Mp`).
    ///
    /// The occlusion counterpart of [`intersect1mp`](Self::intersect1mp); an
    /// occluded ray has its `tfar` set to `-inf` in place.
    pub fn occluded1mp<C: AsIntersectContext>(&self, ctx: &mut C, rays: &mut [&mut Ray]) {
        let mut ptrs: Vec<*mut RTCRay> = rays.iter_mut().map(|r| &mut **r as *mut RTCRay).collect();
        unsafe {
            rtcOccluded1Mp(
                self.handle,
                ctx.as_mut_context_ptr(),
                ptrs.as_mut_ptr(),
                ptrs.len() as u32,
            );
        }
    }

    /// Reports candidate colliding primitive pairs between this scene and
    /// `other`.
    ///
    /// embree traverses the two scenes' BVHs and invokes `callback` with
    /// batches of [`Collision`]s. This is a **broad phase**: each reported
    /// pair is a potentially-intersecting primitive pair from a leaf pair
    /// reached during traversal. embree does **not** test the individual
    /// primitive (or even leaf) bounds before reporting, so a reported
    /// pair's bounds need not overlap. The callback must do the
    /// narrow-phase primitive/primitive test itself to reject false
    /// positives, and accumulate results into its captured state. The batch
    /// slice is `&mut` so the callback may compact it in place (e.g. keep
    /// only the surviving pairs) as scratch; embree ignores the slice after
    /// the call. Pass the same scene as both `self` and `other`
    /// for self-collision; embree omits the `(primitive, same primitive)` pair
    /// but does **not** guarantee ordering or uniqueness (it may report both
    /// `(A, B)` and `(B, A)`).
    ///
    /// The callback runs on multiple worker threads, so `F` is `Fn + Sync` and
    /// any captured state must be shared safely (e.g. atomics or a `Mutex`).
    /// Each invocation gets its own batch, so the `&mut` slice never aliases
    /// across threads.
    ///
    /// # Errors
    ///
    /// embree validates the preconditions only in debug builds, and even then
    /// incompletely (its geometry check uses `&&`, so it misses the case where
    /// only *one* scene is invalid), so this wrapper checks the cheap ones up
    /// front and returns [`Error::INVALID_ARGUMENT`] **before** calling embree
    /// when:
    /// - the two scenes were created on **different devices**;
    /// - either scene has **no attached geometry** (an empty scene is an
    ///   `AccelN`, but the collider casts both scenes' acceleration structures
    ///   to a single `BVH`); or
    /// - either scene contains a **non-user geometry** (the collider only
    ///   understands user geometries).
    ///
    /// # Safety
    ///
    /// The checks above do not cover everything embree requires. The caller
    /// must still guarantee all of:
    /// - both scenes are **committed** (since their last modification);
    /// - each geometry has a **single time step** (not checked here);
    /// - each scene has **at least one enabled, non-degenerate primitive** (the
    ///   `# Errors` check only rejects a scene with *no* geometry, not one
    ///   whose geometries are all disabled or zero-primitive);
    /// - both use the **same BVH layout** -- in particular the same `COMPACT`
    ///   scene flag. embree picks one collider (`BVH4` vs `BVH8`, which differ
    ///   on AVX hardware) from `self` and applies it to *both* scenes' nodes,
    ///   so a mismatch reinterprets the other scene's nodes at the wrong width;
    /// - the `callback` must **not** mutate, commit, attach to, or detach from
    ///   either scene (or their geometries) while `collide` is running.
    ///
    /// Violating any of these is undefined behavior in a release embree. (A
    /// fully safe wrapper would have to also track per-scene committed state,
    /// time-step counts, per-geometry enablement, and BVH layout; until then
    /// this is `unsafe`.)
    pub unsafe fn collide<F>(&self, other: &Scene<'a>, callback: F) -> Result<(), Error>
    where
        F: Fn(&mut [Collision]) + Sync,
    {
        // Cheap, safe preconditions the wrapper can check without tracking extra
        // state. The remaining ones stay in the `# Safety` contract above.
        if self.device.handle != other.device.handle {
            return Err(Error::INVALID_ARGUMENT);
        }
        if self.geometries.is_empty() || other.geometries.is_empty() {
            return Err(Error::INVALID_ARGUMENT);
        }
        if self
            .geometries
            .values()
            .chain(other.geometries.values())
            .any(|g| g.shared.kind != GeometryKind::USER)
        {
            return Err(Error::INVALID_ARGUMENT);
        }

        // The callback fires on embree's worker threads and `rtcCollide` blocks until
        // done, so `callback` only needs to live for this call: pass a pointer
        // to it directly (no heap ownership). The trampoline is monomorphized
        // over `F`.
        unsafe extern "C" fn trampoline<F>(
            user_ptr: *mut std::os::raw::c_void,
            collisions: *mut RTCCollision,
            num: std::os::raw::c_uint,
        ) where
            F: Fn(&mut [Collision]) + Sync,
        {
            // Never unwind across the FFI boundary: a panic on a worker thread aborts.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let cb = &*(user_ptr as *const F);
                // Each callback owns a distinct batch array, so a `&mut` view is sound
                // even though `cb` is shared across threads. The callback may scribble
                // on it (compact in place); embree discards it afterwards.
                let pairs =
                    std::slice::from_raw_parts_mut(collisions as *mut Collision, num as usize);
                cb(pairs);
            }));
            if result.is_err() {
                std::process::abort();
            }
        }

        unsafe {
            rtcCollide(
                self.handle,
                other.handle,
                Some(trampoline::<F>),
                &callback as *const F as *mut std::os::raw::c_void,
            );
        }
        Ok(())
    }

    /// Returns the axis-aligned bounding box of the scene.
    pub fn get_bounds(&self) -> Bounds {
        let mut bounds = Bounds {
            lower_x: 0.0,
            upper_x: 0.0,
            lower_y: 0.0,
            upper_y: 0.0,
            lower_z: 0.0,
            upper_z: 0.0,
            align0: 0.0,
            align1: 0.0,
        };
        unsafe {
            rtcGetSceneBounds(self.handle(), &mut bounds as *mut Bounds);
        }
        bounds
    }

    /// Returns the scene's linear (motion-blur) bounds: the axis-aligned
    /// bounding box at the start and end of the time range (`bounds0` /
    /// `bounds1`).
    ///
    /// For a scene without motion blur, both equal the static
    /// [`get_bounds`](Self::get_bounds). May only be called after
    /// [`commit`](Self::commit).
    pub fn get_linear_bounds(&self) -> LinearBounds {
        let zero = Bounds {
            lower_x: 0.0,
            upper_x: 0.0,
            lower_y: 0.0,
            upper_y: 0.0,
            lower_z: 0.0,
            upper_z: 0.0,
            align0: 0.0,
            align1: 0.0,
        };
        let mut bounds = LinearBounds {
            bounds0: zero,
            bounds1: zero,
        };
        unsafe {
            rtcGetSceneLinearBounds(self.handle, &mut bounds as *mut LinearBounds);
        }
        bounds
    }
}

/// User data for the packet `point_query4/8/16` callbacks: a shared closure
/// pointer plus the lane's own `data` accumulator. The cast back to the
/// concrete `F`/`D` is sound by construction (the trampoline is monomorphized
/// over the same types the binder used), so no runtime `TypeId` guard is
/// needed.
#[derive(Debug)]
pub(crate) struct PointQueryCallbackData {
    pub scene_closure: *mut std::os::raw::c_void,
    pub data: *mut std::os::raw::c_void,
}

/// Helper function to convert a Rust closure to `RTCProgressMonitorFunction`
/// callback.
fn progress_monitor_function<F>() -> RTCProgressMonitorFunction
where
    F: Fn(f64) -> bool + Send + Sync + 'static,
{
    unsafe extern "C" fn inner<F>(f: *mut std::os::raw::c_void, n: f64) -> bool
    where
        F: Fn(f64) -> bool + Send + Sync + 'static,
    {
        let cb = &*(f as *const F);
        cb(n)
    }

    Some(inner::<F>)
}

/// Helper function to convert a Rust closure to `RTCPointQueryFunction`
/// callback.
fn point_query_function<F>() -> RTCPointQueryFunction
where
    F: FnMut(&mut PointQuery, &mut PointQueryContext, u32, u32, f32) -> bool,
{
    unsafe extern "C" fn inner<F>(args: *mut RTCPointQueryFunctionArguments) -> bool
    where
        F: FnMut(&mut PointQuery, &mut PointQueryContext, u32, u32, f32) -> bool,
    {
        // For the single query, `userPtr` is the closure itself (it captures its
        // own state, so there is no separate callback record).
        let cb = &mut *((*args).userPtr as *mut F);
        cb(
            &mut *(*args).query,
            &mut *(*args).context,
            (*args).primID,
            (*args).geomID,
            (*args).similarityScale,
        )
    }

    Some(inner::<F>)
}

/// Packet-query variant of [`point_query_function`]: each lane's `userPtr`
/// points at its own [`PointQueryCallbackData`] (shared closure pointer,
/// per-lane `data`), so the closure receives the lane's own `&mut D`.
fn point_query_function_n<F, D>() -> RTCPointQueryFunction
where
    F: FnMut(&mut PointQuery, &mut PointQueryContext, &mut D, u32, u32, f32) -> bool,
{
    unsafe extern "C" fn inner<F, D>(args: *mut RTCPointQueryFunctionArguments) -> bool
    where
        F: FnMut(&mut PointQuery, &mut PointQueryContext, &mut D, u32, u32, f32) -> bool,
    {
        let cbdata = &mut *((*args).userPtr as *mut PointQueryCallbackData);
        // `scene_closure` (shared `F`) and `data` (this lane's `D`) are always
        // bound by `point_query{4,8,16}`; the casts are sound by monomorphization.
        let cb = &mut *(cbdata.scene_closure as *mut F);
        cb(
            &mut *(*args).query,
            &mut *(*args).context,
            &mut *(cbdata.data as *mut D),
            (*args).primID,
            (*args).geomID,
            (*args).similarityScale,
        )
    }

    Some(inner::<F, D>)
}

#[cfg(test)]
mod valid_mask_tests {
    //! `ValidMaskN` must be 16-byte aligned so it reaches embree without a
    //! copy. Pure-Rust.
    use super::ValidMaskN;

    #[test]
    fn valid_mask_is_16_byte_aligned() {
        assert_eq!(std::mem::align_of::<ValidMaskN<4>>(), 16);
        assert_eq!(std::mem::align_of::<ValidMaskN<8>>(), 16);
        assert_eq!(std::mem::align_of::<ValidMaskN<16>>(), 16);
        let m = ValidMaskN::<8>::all_active();
        assert_eq!(m.as_ptr() as usize % 16, 0, "mask must be 16-aligned");
    }
}
