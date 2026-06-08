//! Safe wrapper over embree's standalone BVH builder (`rtcBuildBVH` +
//! `rtcThreadLocalAlloc`).
//!
//! Unlike the scene BVH that `rtcCommitScene` builds implicitly, this is a
//! *bring-your-own-node-layout* builder: embree decides the tree topology and
//! the build heuristic, and for every node it produces it calls back to
//! allocate and fill the caller's own node type in embree's arena. The caller
//! therefore owns the in-memory layout and its own traversal; embree owns only
//! the structure.
//!
//! # Entry point
//!
//! [`Device::create_bvh`](crate::Device::create_bvh) returns a [`Bvh`]; a build
//! runs through [`Bvh::build_scoped`], driven by a caller-supplied
//! [`BvhBuilder`]. The resulting tree is navigated inside the build scope
//! through [`BvhResult`].
//!
//! # Design notes
//!
//! - **Generative scope.** `build_scoped` takes a `for<'id>
//!   FnOnce(BvhResult<'id, B>)` closure, so the branded lifetime `'id` is
//!   unique per call: a [`NodePtr`] handle cannot escape the closure or be
//!   resolved against a different build, and a handle becomes a `&Node` only
//!   through [`BvhResult::resolve`]. A plain invariant lifetime would not be
//!   enough: two independent builds could be inferred to share one lifetime,
//!   letting a handle from one resolve against the other's arena; the
//!   `for<'id>` bound makes `'id` fresh per call so the compiler cannot unify
//!   two builds.
//!
//! - **Arena memory.** Nodes are bump-allocated through [`Allocator`]
//!   (`rtcThreadLocalAlloc`) and freed wholesale when the [`Bvh`] drops;
//!   **`Drop` never runs on a node**. The [`BvhNode`] bound (`Copy + Send +
//!   Sync`, no owned resources) is what makes that sound. Child links are
//!   opaque [`NodePtr`] handles, never live references held during the build.
//!
//! - **Threaded callbacks.** Embree invokes the [`BvhBuilder`] methods from
//!   many worker threads (hence the `Send + Sync` bound). The `extern "C"`
//!   trampolines recover the builder from a raw `userPtr` (`BuildState`) and
//!   rebrand the callback-local pointers to the result's `'id`
//!   (`build_with_brand`); the soundness invariant is that every pointer in one
//!   callback belongs to the build identified by that callback's `userPtr`.
//!
//! - **Version-gated.** The safe path relies on a per-node setter-serialization
//!   property audited for embree 3.13.5, checked at runtime against the linked
//!   library's version.
//!
//! See [`BvhBuilder`] for the callback contract and a worked example.
use std::{
    marker::PhantomData,
    os::raw::{c_uint, c_void},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::NonNull,
};

use crate::{sys::*, Bounds, BuildPrimitive, BuildQuality, Device, DeviceProperty, Error};

/// A reference-counted standalone BVH (`RTCBVH`). Build target for
/// [`Bvh::build_scoped`].
///
/// Not `Clone`: the result of a build borrows the `Bvh` exclusively, and a
/// second handle could rebuild and free the arena out from under a live result.
pub struct Bvh {
    device: Device,
    handle: RTCBVH,
}

impl Drop for Bvh {
    fn drop(&mut self) { unsafe { rtcReleaseBVH(self.handle) } }
}

impl Bvh {
    pub(crate) fn new(device: &Device) -> Result<Self, Error> {
        let handle = unsafe { rtcNewBVH(device.handle) };
        if handle.is_null() {
            Err(device.get_error())
        } else {
            Ok(Self {
                device: device.clone(),
                handle,
            })
        }
    }
}

/// Marker for a type that may live in the BVH arena.
///
/// # Safety
///
/// The arena frees node memory in bulk without running `Drop`, and embree
/// creates/mutates nodes across worker threads. Implementors must own no
/// external resources (a raw pointer to an owned heap allocation would leak; an
/// embedded foreign reference could dangle) and must be sound to construct and
/// mutate across threads. A node should contain only plain `Copy` data,
/// [`Bounds`], and [`NodePtr`] handles from the same build.
///
/// **Family-level rebrand requirement.** This trait is implemented across all
/// lifetimes (`unsafe impl<'id> BvhNode for MyNode<'id>`), and that blanket
/// impl is what makes the builder's *rebrand* sound: nodes are allocated at a
/// callback-local lifetime and the wrapper later reinterprets the root as
/// [`BvhBuilder::Node`]`<'id>` for the result's brand. For any lifetimes `'a`
/// and `'b`, `MyNode<'a>` and `MyNode<'b>` must therefore be **layout-identical
/// and safely rebrandable**. That holds automatically for any type meeting the
/// rule above: Rust layout never depends on a lifetime, and the only permitted
/// lifetime-carrying members are `NodePtr` phantom brands (which rebrand for
/// free). Do not embed any other lifetime-dependent state.
pub unsafe trait BvhNode: Copy + Send + Sync {}

/// Opaque, generatively branded handle to an arena node. `Copy`, pointer-sized,
/// and not dereferenceable except through [`BvhResult::resolve`] inside the
/// build scope.
#[repr(transparent)]
pub struct NodePtr<'id, N> {
    ptr: NonNull<N>,
    _brand: PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id, N> Clone for NodePtr<'id, N> {
    fn clone(&self) -> Self { *self }
}
impl<'id, N> Copy for NodePtr<'id, N> {}

// Deliberate: an opaque non-dereferenceable handle is safe to move/share (the
// brand still confines it to the scope).
unsafe impl<'id, N> Send for NodePtr<'id, N> {}
unsafe impl<'id, N> Sync for NodePtr<'id, N> {}

/// Per-callback-thread bump allocator over the BVH arena
/// (`rtcThreadLocalAlloc`). `!Send + !Sync`: valid only on its current callback
/// thread.
pub struct Allocator<'id> {
    raw: RTCThreadLocalAllocator,
    _brand: PhantomData<&'id ()>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'id> Allocator<'id> {
    unsafe fn from_raw(raw: RTCThreadLocalAllocator) -> Self {
        Self {
            raw,
            _brand: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Bump-allocate one `T` in the arena and move `value` into it.
    ///
    /// ZSTs are rejected **unconditionally** (also in release): embree relies
    /// on node-pointer identity, and a zero-size allocation would alias
    /// every node to one address. This runs behind FFI, so it aborts rather
    /// than unwinds. (`build_with_brand` also rejects a ZST `Node` up front
    /// with a clean `Err`; this is the defensive backstop for any other
    /// `alloc::<T>` call.)
    pub fn alloc<T: Copy>(&self, value: T) -> &'id mut T {
        if std::mem::size_of::<T>() == 0 {
            std::process::abort();
        }
        let layout = std::alloc::Layout::new::<T>();
        let p = unsafe { rtcThreadLocalAlloc(self.raw, layout.size(), layout.align()) } as *mut T;
        if p.is_null() {
            // No unwinding across FFI: OOM aborts.
            std::process::abort();
        }
        unsafe {
            p.write(value);
            &mut *p
        }
    }
}

/// Checked, view-scoped accessor over embree's `void**` child array.
pub struct Children<'id, N> {
    ptr: *const *mut c_void,
    len: usize,
    _m: PhantomData<NodePtr<'id, N>>,
}

impl<'id, N> Children<'id, N> {
    unsafe fn from_raw(ptr: *mut *mut c_void, len: usize) -> Self {
        Self {
            ptr: ptr as *const *mut c_void,
            len,
            _m: PhantomData,
        }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// The `i`th child handle, or `None` if out of range.
    pub fn get(&self, i: usize) -> Option<NodePtr<'id, N>> {
        if i >= self.len {
            return None;
        }
        let raw = unsafe { *self.ptr.add(i) };
        NonNull::new(raw as *mut N).map(|ptr| NodePtr {
            ptr,
            _brand: PhantomData,
        })
    }
}

/// Checked accessor over embree's `RTCBounds*` array. References borrow the
/// view (the bounds may live in callback-local storage).
pub struct ChildBounds<'a> {
    ptr: *const *const RTCBounds,
    len: usize,
    _m: PhantomData<&'a Bounds>,
}

impl<'a> ChildBounds<'a> {
    unsafe fn from_raw(ptr: *mut *const RTCBounds, len: usize) -> Self {
        Self {
            ptr: ptr as *const *const RTCBounds,
            len,
            _m: PhantomData,
        }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// The `i`th child's bounds, borrowed from this view, or `None` if out of
    /// range.
    pub fn get(&self, i: usize) -> Option<&Bounds> {
        if i >= self.len {
            return None;
        }
        unsafe { (*self.ptr.add(i)).as_ref() }
    }
}

/// Materializes embree's BVH topology into a caller-defined node layout.
///
/// [`Bvh::build_scoped`] runs embree's builder, which decides the tree's shape
/// and, for every node it produces, calls back into this trait to allocate and
/// fill *your* node type in embree's arena. You own the in-memory layout (and
/// therefore your own traversal); embree owns only the topology and the build
/// heuristic. This is the safe wrapper over embree's `rtcBuildBVH` callback
/// API.
///
/// # Callback lifecycle
///
/// For each interior node, embree calls (in this order, serialized per node):
/// 1. [`create_node`](Self::create_node) - allocate the node; *its children do
///    not exist yet*.
/// 2. (embree recurses and builds the children.)
/// 3. [`set_bounds`](Self::set_bounds) - store each child's bounding box into
///    the node.
/// 4. [`set_children`](Self::set_children) - store each child's handle into the
///    node.
///
/// For each leaf, embree calls [`create_leaf`](Self::create_leaf) with the
/// primitives that fall into it. When [`SPATIAL_SPLITS`](Self::SPATIAL_SPLITS)
/// is enabled (and quality is `HIGH`), [`split`](Self::split) may be called to
/// divide a primitive across a plane; when [`PROGRESS`](Self::PROGRESS) is
/// enabled, [`progress`](Self::progress) is called to report.
///
/// # Threading
///
/// Embree invokes these callbacks from **many worker threads**, so `Self` is
/// `Send + Sync` and every method takes a shared `&self`. Any per-build mutable
/// state must therefore live behind atomics or locks, not `&mut self`. (Embree
/// does not invoke two callbacks for the *same* node concurrently; this is the
/// one audited assumption noted in the crate docs.)
///
/// # Memory
///
/// Nodes are bump-allocated in embree's arena via the [`Allocator`] handed to
/// `create_node`/`create_leaf`, and the whole arena is freed at once when the
/// [`Bvh`] is dropped - **`Drop` never runs on a node**. The [`BvhNode`] bound
/// enforces what that requires (`Copy`, `Send + Sync`, no owned resources).
/// Child links are opaque, scope-branded [`NodePtr`] handles; you can only turn
/// one back into a `&Node` after the build, through [`BvhResult::resolve`].
///
/// # Example
///
/// ```no_run
/// use embree3::{
///     Allocator, Bounds, BuildConfig, BuildPrimitive, BvhBuilder, BvhNode, ChildBounds, Children,
///     Device, NodePtr,
/// };
///
/// const EMPTY: Bounds = Bounds {
///     lower_x: 0.0,
///     lower_y: 0.0,
///     lower_z: 0.0,
///     align0: 0.0,
///     upper_x: 0.0,
///     upper_y: 0.0,
///     upper_z: 0.0,
///     align1: 0.0,
/// };
///
/// // A single, generatively branded node type (one constant `N` sizes both the child
/// // array and `MAX_CHILDREN`). Inner children are `Option` because they are filled in
/// // `set_children`, after `create_node` has already returned the node.
/// const N: usize = 2;
/// #[derive(Clone, Copy)]
/// enum Node<'id> {
///     Inner {
///         bounds: [Bounds; N],
///         kids: [Option<NodePtr<'id, Node<'id>>>; N],
///     },
///     Leaf {
///         prim_id: u32,
///     },
/// }
/// unsafe impl<'id> BvhNode for Node<'id> {}
///
/// struct Builder;
/// impl BvhBuilder for Builder {
///     type Node<'id> = Node<'id>;
///     const MAX_CHILDREN: usize = N;
///
///     fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Node<'id> {
///         a.alloc(Node::Inner {
///             bounds: [EMPTY; N],
///             kids: [None; N],
///         })
///     }
///     fn set_children<'id>(&self, node: &mut Node<'id>, c: Children<'id, Node<'id>>) {
///         if let Node::Inner { kids, .. } = node {
///             for i in 0..c.len().min(N) {
///                 kids[i] = c.get(i);
///             }
///         }
///     }
///     fn set_bounds<'id>(&self, node: &mut Node<'id>, b: ChildBounds<'_>) {
///         if let Node::Inner { bounds, .. } = node {
///             for i in 0..b.len().min(N) {
///                 if let Some(cb) = b.get(i) {
///                     bounds[i] = *cb;
///                 }
///             }
///         }
///     }
///     fn create_leaf<'id>(
///         &self,
///         a: &Allocator<'id>,
///         prims: &[BuildPrimitive],
///     ) -> &'id mut Node<'id> {
///         // With `max_leaf_size = 1` each leaf holds one primitive.
///         a.alloc(Node::Leaf {
///             prim_id: prims[0].primID,
///         })
///     }
/// }
///
/// let device = Device::new().unwrap();
/// let mut bvh = device.create_bvh().unwrap();
/// let mut prims: Vec<BuildPrimitive> = Vec::new(); // fill with your primitive AABBs
/// let cfg = BuildConfig {
///     max_leaf_size: 1,
///     ..Default::default()
/// };
///
/// // Navigate the tree inside the scope; handles cannot escape `f`.
/// let has_root = bvh
///     .build_scoped(&cfg, &mut prims, &Builder, |r| r.root().is_some())
///     .unwrap();
/// ```
pub trait BvhBuilder: Send + Sync {
    /// The arena node type, generatively branded by the build scope `'id`.
    ///
    /// Almost always an `enum { Inner { .. }, Leaf { .. } }`. Because
    /// [`create_node`](Self::create_node) runs before a node's children exist,
    /// the inner variant stores child handles as `[Option<NodePtr<'id,
    /// Self::Node<'id>>>; N]` (filled in
    /// [`set_children`](Self::set_children)); `Option<NodePtr>` is
    /// niche-optimized to a single pointer, so the `Option` costs nothing.
    /// `N` is a concrete constant that must equal
    /// [`MAX_CHILDREN`](Self::MAX_CHILDREN).
    ///
    /// The `+ 'id` bound makes `&'id Node<'id>` well-formed; see [`BvhNode`]
    /// for the family-level rebrand requirement the builder relies on.
    type Node<'id>: BvhNode + 'id;

    /// Maximum number of children an inner node can hold, i.e. the length of
    /// the node's child array.
    ///
    /// [`Bvh::build_scoped`] rejects a `BuildConfig::max_branching_factor`
    /// greater than this, so [`set_children`](Self::set_children) can never
    /// receive more children than the array holds.
    const MAX_CHILDREN: usize;

    /// Enable spatial splits. When `true`, the [`split`](Self::split) callback
    /// is registered and embree may split primitives across node boundaries
    /// (only at `BuildQuality::HIGH`); [`Bvh::build_scoped`] reserves the
    /// extra primitive-array capacity this needs. Leaving it `false` (the
    /// default) registers no split callback, so build behavior is unchanged.
    const SPATIAL_SPLITS: bool = false;

    /// Enable progress reporting through [`progress`](Self::progress). Default
    /// `false` (no progress callback is registered).
    const PROGRESS: bool = false;

    /// Allocate an interior node that will have `child_count` children
    /// (`child_count <= `[`MAX_CHILDREN`](Self::MAX_CHILDREN)) and return it.
    ///
    /// Allocate in the arena with [`Allocator::alloc`] and return the
    /// reference. The children do **not** exist yet, so initialize the
    /// child slots to a placeholder (e.g. `[None; N]`);
    /// [`set_bounds`](Self::set_bounds) and
    /// [`set_children`](Self::set_children) fill them once the children
    /// have been built.
    fn create_node<'id>(
        &self,
        alloc: &Allocator<'id>,
        child_count: usize,
    ) -> &'id mut Self::Node<'id>;

    /// Store the handles of `node`'s children into `node`.
    ///
    /// The children were produced earlier by [`create_node`](Self::create_node)
    /// / [`create_leaf`](Self::create_leaf). `children` is a checked,
    /// scope-branded view: `children.get(i)` yields the `i`th child's
    /// [`NodePtr`] (or `None` past the end). Copy the handles into the
    /// node's child array; a handle can only be dereferenced after the
    /// build, through [`BvhResult::resolve`].
    fn set_children<'id>(
        &self,
        node: &mut Self::Node<'id>,
        children: Children<'id, Self::Node<'id>>,
    );

    /// Store the bounding boxes of `node`'s children into `node`.
    ///
    /// `bounds.get(i)` borrows the `i`th child's [`Bounds`] **from the view** -
    /// embree may keep it in callback-local storage, so copy the value out
    /// rather than retaining the reference.
    fn set_bounds<'id>(&self, node: &mut Self::Node<'id>, bounds: ChildBounds<'_>);

    /// Allocate a leaf node over `prims` and return it.
    ///
    /// `prims` is the complete set of primitives in this leaf
    /// (`1 ..= BuildConfig::max_leaf_size`). The leaf must represent **all** of
    /// them (store their `primID`s, an index range, etc.) - keeping only
    /// `prims[0]` silently drops the rest unless you build with
    /// `max_leaf_size == 1`. Allocate via [`Allocator::alloc`]; a
    /// leaf cannot own heap memory (see [`BvhNode`]).
    fn create_leaf<'id>(
        &self,
        alloc: &Allocator<'id>,
        prims: &[BuildPrimitive],
    ) -> &'id mut Self::Node<'id>;

    /// Split a primitive's bounding box by the plane `dim = pos`, returning its
    /// `(left, right)` sub-boxes.
    ///
    /// Only called when [`SPATIAL_SPLITS`](Self::SPATIAL_SPLITS) is `true`.
    /// `dim` is the axis (`0 = x`, `1 = y`, `2 = z`). The default clips the
    /// primitive's box at `pos` along `dim` (a valid, conservative
    /// geometric split); override it for tighter, geometry-aware splits.
    fn split(&self, prim: &BuildPrimitive, dim: u32, pos: f32) -> (Bounds, Bounds) {
        let lo_full = bounds_of(prim);
        let mut lo = lo_full;
        let mut hi = lo_full;
        set_axis_upper(&mut lo, dim, pos);
        set_axis_lower(&mut hi, dim, pos);
        (lo, hi)
    }

    /// Report build progress, with `fraction` advancing toward `1.0`.
    ///
    /// Only called when [`PROGRESS`](Self::PROGRESS) is `true`. **Report
    /// only:** embree 3.13.5's standalone builder discards the callback's
    /// result, so this cannot cancel the build (hence the `()` return
    /// rather than a `bool`).
    fn progress(&self, _fraction: f64) {}
}

/// Build settings. `Default` reproduces `rtcDefaultBuildArguments`.
#[derive(Clone, Debug)]
pub struct BuildConfig {
    pub quality: BuildQuality,
    /// Optimize the build for fast *rebuilds* of dynamic scenes, at the cost of
    /// higher memory use (embree's `RTC_BUILD_FLAG_DYNAMIC`). `false`
    /// builds for best query performance.
    pub dynamic: bool,
    pub max_branching_factor: u32,
    pub max_depth: u32,
    pub sah_block_size: u32,
    pub min_leaf_size: u32,
    pub max_leaf_size: u32,
    pub traversal_cost: f32,
    pub intersection_cost: f32,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            quality: BuildQuality::MEDIUM,
            dynamic: false,
            max_branching_factor: 2,
            max_depth: 32,
            sah_block_size: 1,
            min_leaf_size: 1,
            max_leaf_size: 32,
            traversal_cost: 1.0,
            intersection_cost: 1.0,
        }
    }
}

impl BuildConfig {
    fn validate(&self, prim_count: usize, max_children: usize) -> Result<(), Error> {
        let bad = Err(Error::INVALID_ARGUMENT);
        match self.quality {
            BuildQuality::LOW | BuildQuality::MEDIUM | BuildQuality::HIGH => {}
            _ => return bad,
        }
        let branch_cap = if self.quality == BuildQuality::LOW {
            8
        } else {
            16
        };
        if self.max_branching_factor < 2
            || self.max_branching_factor > branch_cap
            || self.max_branching_factor as usize > max_children
        {
            return bad;
        }
        if self.max_leaf_size < 1 || self.max_leaf_size > 32 {
            return bad;
        }
        if self.min_leaf_size < 1 || self.min_leaf_size > self.max_leaf_size {
            return bad;
        }
        if self.sah_block_size < 1 || self.max_depth < 1 {
            return bad;
        }
        if !self.traversal_cost.is_finite()
            || self.traversal_cost <= 0.0
            || !self.intersection_cost.is_finite()
            || self.intersection_cost <= 0.0
        {
            return bad;
        }
        if self.quality == BuildQuality::LOW && prim_count > u32::MAX as usize {
            return bad;
        }
        Ok(())
    }
}

/// Result of a build, valid only inside the generative scope. Branded by `'id`.
pub struct BvhResult<'id, B: BvhBuilder> {
    root: Option<NodePtr<'id, B::Node<'id>>>,
    _brand: PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id, B: BvhBuilder> BvhResult<'id, B> {
    /// The root node, or `None` for an empty tree (zero primitives).
    pub fn root(&self) -> Option<&B::Node<'id>> { self.root.map(|p| unsafe { p.ptr.as_ref() }) }
    /// The root handle.
    pub fn root_ptr(&self) -> Option<NodePtr<'id, B::Node<'id>>> { self.root }
    /// Resolve a child handle to a node reference. Safe with no runtime check:
    /// the `'id` brand proves the handle came from this same build, and the
    /// arena cannot have been freed or rebuilt -- the generative `build_scoped`
    /// closure that produced this [`BvhResult`] holds the `&mut Bvh`
    /// exclusively for its whole duration, so no rebuild or drop can run
    /// while `'id` is live. (The returned reference borrows `self`, tying
    /// it to the result handle.)
    pub fn resolve(&self, p: NodePtr<'id, B::Node<'id>>) -> &B::Node<'id> {
        unsafe { p.ptr.as_ref() }
    }
}

/// Internal state passed through `userPtr`. The builder is held as a raw
/// pointer (valid for the synchronous `rtcBuildBVH` call by the rebrand
/// invariant), which avoids a `B: 'x` outlives requirement when a trampoline
/// recovers it at a fresh lifetime. The generative brand lives only on
/// [`BvhResult`], not here.
struct BuildState<B> {
    builder: *const B,
}

fn catch_abort<R>(f: impl FnOnce() -> R) -> R {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => std::process::abort(),
    }
}

// Trampolines.
// Each recovers `&BuildState` then rebrands all pointers at one inferred
// lifetime 'x (the unsafe invariant: every pointer in one synchronous callback
// belongs to the in-flight build whose BuildState is `userPtr`).

// Recover the builder reference from `userPtr`. The returned `&B` lives for an
// inferred lifetime; soundness is the rebrand invariant (the pointer belongs to
// the in-flight build).
#[inline]
unsafe fn builder_of<'x, B: BvhBuilder>(user: *mut c_void) -> &'x B {
    &*((*(user as *const BuildState<B>)).builder)
}

unsafe extern "C" fn create_node_tramp<B: BvhBuilder>(
    alloc: RTCThreadLocalAllocator,
    child_count: c_uint,
    user: *mut c_void,
) -> *mut c_void {
    catch_abort(|| unsafe {
        let builder = builder_of::<B>(user);
        let allocator = Allocator::from_raw(alloc);
        let node = builder.create_node(&allocator, child_count as usize);
        node as *mut _ as *mut c_void
    })
}

unsafe extern "C" fn create_leaf_tramp<B: BvhBuilder>(
    alloc: RTCThreadLocalAllocator,
    prims: *const RTCBuildPrimitive,
    prim_count: usize,
    user: *mut c_void,
) -> *mut c_void {
    catch_abort(|| unsafe {
        let builder = builder_of::<B>(user);
        let allocator = Allocator::from_raw(alloc);
        let slice = std::slice::from_raw_parts(prims, prim_count);
        let node = builder.create_leaf(&allocator, slice);
        node as *mut _ as *mut c_void
    })
}

unsafe extern "C" fn set_children_tramp<B: BvhBuilder>(
    node: *mut c_void,
    children: *mut *mut c_void,
    child_count: c_uint,
    user: *mut c_void,
) {
    catch_abort(|| unsafe {
        let builder = builder_of::<B>(user);
        let node = &mut *(node as *mut B::Node<'_>);
        let kids = Children::from_raw(children, child_count as usize);
        builder.set_children(node, kids);
    })
}

unsafe extern "C" fn set_bounds_tramp<B: BvhBuilder>(
    node: *mut c_void,
    bounds: *mut *const RTCBounds,
    child_count: c_uint,
    user: *mut c_void,
) {
    catch_abort(|| unsafe {
        let builder = builder_of::<B>(user);
        let node = &mut *(node as *mut B::Node<'_>);
        let view = ChildBounds::from_raw(bounds, child_count as usize);
        builder.set_bounds(node, view);
    })
}

unsafe extern "C" fn split_tramp<B: BvhBuilder>(
    prim: *const RTCBuildPrimitive,
    dim: c_uint,
    pos: f32,
    lbounds: *mut RTCBounds,
    rbounds: *mut RTCBounds,
    user: *mut c_void,
) {
    catch_abort(|| unsafe {
        let builder = builder_of::<B>(user);
        let (lo, hi) = builder.split(&*prim, dim, pos);
        *lbounds = lo;
        *rbounds = hi;
    });
}

unsafe extern "C" fn progress_tramp<B: BvhBuilder>(user: *mut c_void, n: f64) -> bool {
    catch_abort(|| unsafe {
        let builder = builder_of::<B>(user);
        builder.progress(n);
    });
    true
}

impl Bvh {
    /// Build a BVH over `primitives` inside a generative scope. The result and
    /// any [`NodePtr`] handles cannot escape `f`; return owned data
    /// (counts, a `Vec`, a copied-out tree) instead.
    ///
    /// The generative brand makes escaping a handle a compile error, which is
    /// what prevents resolving a handle against a different build:
    ///
    /// ```compile_fail
    /// use embree3::{
    ///     Allocator, Bvh, BvhBuilder, BvhNode, BuildConfig, BuildPrimitive, ChildBounds,
    ///     Children, Device,
    /// };
    /// #[derive(Clone, Copy)]
    /// struct Node {
    ///     n: u32,
    /// }
    /// unsafe impl BvhNode for Node {}
    /// struct B;
    /// impl BvhBuilder for B {
    ///     type Node<'id> = Node;
    ///     const MAX_CHILDREN: usize = 2;
    ///     fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Node {
    ///         a.alloc(Node { n: 0 })
    ///     }
    ///     fn set_children<'id>(&self, _n: &mut Node, _c: Children<'id, Node>) {}
    ///     fn set_bounds<'id>(&self, _n: &mut Node, _b: ChildBounds<'_>) {}
    ///     fn create_leaf<'id>(&self, a: &Allocator<'id>, _p: &[BuildPrimitive]) -> &'id mut Node {
    ///         a.alloc(Node { n: 0 })
    ///     }
    /// }
    /// let device = Device::new().unwrap();
    /// let mut bvh = device.create_bvh().unwrap();
    /// let mut prims: Vec<BuildPrimitive> = Vec::new();
    /// // ERROR: a branded handle cannot escape the generative scope.
    /// let _escaped = bvh.build_scoped(&BuildConfig::default(), &mut prims, &B, |r| r.root_ptr());
    /// ```
    ///
    /// The cross-build case directly: a handle from one build cannot be
    /// resolved by a nested, independent build (their generative brands
    /// differ):
    ///
    /// ```compile_fail
    /// use embree3::{
    ///     Allocator, Bvh, BvhBuilder, BvhNode, BuildConfig, BuildPrimitive, ChildBounds,
    ///     Children, Device,
    /// };
    /// #[derive(Clone, Copy)]
    /// struct Node {
    ///     n: u32,
    /// }
    /// unsafe impl BvhNode for Node {}
    /// struct B;
    /// impl BvhBuilder for B {
    ///     type Node<'id> = Node;
    ///     const MAX_CHILDREN: usize = 2;
    ///     fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Node {
    ///         a.alloc(Node { n: 0 })
    ///     }
    ///     fn set_children<'id>(&self, _n: &mut Node, _c: Children<'id, Node>) {}
    ///     fn set_bounds<'id>(&self, _n: &mut Node, _b: ChildBounds<'_>) {}
    ///     fn create_leaf<'id>(&self, a: &Allocator<'id>, _p: &[BuildPrimitive]) -> &'id mut Node {
    ///         a.alloc(Node { n: 0 })
    ///     }
    /// }
    /// let device = Device::new().unwrap();
    /// let mut outer = device.create_bvh().unwrap();
    /// let mut inner = device.create_bvh().unwrap();
    /// let mut po: Vec<BuildPrimitive> = Vec::new();
    /// let mut pi: Vec<BuildPrimitive> = Vec::new();
    /// let cfg = BuildConfig::default();
    /// outer.build_scoped(&cfg, &mut po, &B, |ra| {
    ///     // ERROR: `ra`'s handle has a different brand than `rb`'s scope, so it cannot be
    ///     // passed to `rb.resolve`. (Returns a `Copy` field to isolate that one error.)
    ///     inner.build_scoped(&cfg, &mut pi, &B, |rb| {
    ///         rb.resolve(ra.root_ptr().unwrap()).n
    ///     })
    /// }).unwrap().unwrap();
    /// ```
    pub fn build_scoped<B, F, R>(
        &mut self,
        config: &BuildConfig,
        primitives: &mut Vec<BuildPrimitive>,
        builder: &B,
        f: F,
    ) -> Result<R, Error>
    where
        B: BvhBuilder,
        F: for<'id> FnOnce(BvhResult<'id, B>) -> R,
    {
        self.build_with_brand(config, primitives, builder, f)
    }

    fn build_with_brand<'id, B, F, R>(
        &mut self,
        config: &BuildConfig,
        primitives: &mut Vec<BuildPrimitive>,
        builder: &B,
        f: F,
    ) -> Result<R, Error>
    where
        B: BvhBuilder,
        F: FnOnce(BvhResult<'id, B>) -> R,
    {
        config.validate(primitives.len(), B::MAX_CHILDREN)?;
        if std::mem::size_of::<B::Node<'id>>() == 0 {
            return Err(Error::INVALID_ARGUMENT);
        }
        // Version check (NOT hard enforcement): the safe build path relies on the
        // per-node setter serialization audited for embree 3.13.5. This rejects any
        // other *reported* version, but is a version check plus a
        // trusted-native-library assumption -- a patched libembree3 reporting
        // 31305 with different callback ordering is not detected; hard
        // enforcement would need a vendored/audited binary.
        if self.device.get_property(DeviceProperty::VERSION)? != RTC_VERSION as isize {
            return Err(Error::INVALID_OPERATION);
        }

        if primitives.is_empty() {
            return Ok(f(BvhResult {
                root: None,
                _brand: PhantomData,
            }));
        }

        // Capacity for spatial splits.
        if B::SPATIAL_SPLITS && config.quality == BuildQuality::HIGH {
            let need = primitives
                .len()
                .checked_mul(2)
                .ok_or(Error::INVALID_ARGUMENT)?;
            primitives.reserve(need - primitives.len());
        }
        let capacity = primitives.capacity();
        let count = primitives.len();

        let state = BuildState::<B> {
            builder: builder as *const B,
        };

        let mut args: RTCBuildArguments = unsafe { std::mem::zeroed() };
        args.byteSize = std::mem::size_of::<RTCBuildArguments>();
        args.buildQuality = config.quality;
        args.buildFlags = if config.dynamic {
            RTCBuildFlags::DYNAMIC
        } else {
            RTCBuildFlags::NONE
        };
        args.maxBranchingFactor = config.max_branching_factor;
        args.maxDepth = config.max_depth;
        args.sahBlockSize = config.sah_block_size;
        args.minLeafSize = config.min_leaf_size;
        args.maxLeafSize = config.max_leaf_size;
        args.traversalCost = config.traversal_cost;
        args.intersectionCost = config.intersection_cost;
        args.bvh = self.handle;
        args.primitives = primitives.as_mut_ptr();
        args.primitiveCount = count;
        args.primitiveArrayCapacity = capacity;
        args.createNode = Some(create_node_tramp::<B>);
        args.setNodeChildren = Some(set_children_tramp::<B>);
        args.setNodeBounds = Some(set_bounds_tramp::<B>);
        args.createLeaf = Some(create_leaf_tramp::<B>);
        args.splitPrimitive = if B::SPATIAL_SPLITS {
            Some(split_tramp::<B>)
        } else {
            None
        };
        args.buildProgress = if B::PROGRESS {
            Some(progress_tramp::<B>)
        } else {
            None
        };
        args.userPtr = &state as *const BuildState<B> as *mut c_void;

        let _ = self.device.get_error();
        let root = unsafe { rtcBuildBVH(&args) };
        if root.is_null() {
            return Err(match self.device.get_error() {
                Error::NONE => Error::UNKNOWN,
                e => e,
            });
        }
        // SAFETY (rebrand 2): the callbacks allocated nodes at a callback-local
        // lifetime; here the root is reinterpreted as `B::Node<'id>` for the
        // result's brand. Sound by the `BvhNode` family-level rebrand
        // requirement (all `Node<'_>` are layout-identical and differ only by
        // phantom `NodePtr` brands), and the arena outlives `'id`.
        let root = NonNull::new(root as *mut B::Node<'id>).map(|ptr| NodePtr {
            ptr,
            _brand: PhantomData,
        });
        Ok(f(BvhResult {
            root,
            _brand: PhantomData,
        }))
    }
}

fn bounds_of(p: &BuildPrimitive) -> Bounds {
    Bounds {
        lower_x: p.lower_x,
        lower_y: p.lower_y,
        lower_z: p.lower_z,
        align0: 0.0,
        upper_x: p.upper_x,
        upper_y: p.upper_y,
        upper_z: p.upper_z,
        align1: 0.0,
    }
}
fn set_axis_upper(b: &mut Bounds, dim: u32, pos: f32) {
    match dim {
        0 => b.upper_x = pos,
        1 => b.upper_y = pos,
        _ => b.upper_z = pos,
    }
}
fn set_axis_lower(b: &mut Bounds, dim: u32, pos: f32) {
    match dim {
        0 => b.lower_x = pos,
        1 => b.lower_y = pos,
        _ => b.lower_z = pos,
    }
}
