//! [![Crates.io](https://img.shields.io/crates/v/embree3.svg)](https://crates.io/crates/embree3)
//! [![CI](https://github.com/matthiascy/embree3/actions/workflows/main.yml/badge.svg)](https://github.com/matthiascy/embree3/actions/workflows/main.yml)
//!
//! Safe Rust bindings to [Embree](https://embree.github.io/) 3.13.5, Intel's
//! high-performance ray-tracing kernels.
//!
//! This crate is a thin `unsafe` FFI wrapper whose one job is to turn Embree's
//! raw pointers and C callbacks into a memory-safe Rust API: callback closures
//! are heap-owned and reached through a lock-free per-geometry table; geometry
//! mutation is gated by the [`GeometryBuilder`] / [`Geometry`] typestate; and
//! [`Scene`] is a non-`Clone` unique owner of its handle (share a committed one
//! across threads with `Arc<Scene>`).
//!
//! # Overview
//!
//! - [`Device`] is the entry point. From it you create a [`Scene`], a geometry
//!   builder ([`Device::create_geometry`]), or a standalone BVH
//!   ([`Device::create_bvh`]).
//! - Configure a [`GeometryBuilder`] (buffers, callbacks), [`commit`] it to a
//!   shareable read-only [`Geometry`], attach it to a [`Scene`], and commit the
//!   scene.
//! - Query the scene with [`Scene::intersect`] / [`Scene::occluded`] (single
//!   rays), their `4` / `8` / `16`-wide packet variants, the stream APIs, or
//!   [`Scene::point_query`]; find scene-vs-scene candidate pairs with
//!   [`Scene::collide`].
//!
//! [`commit`]: GeometryBuilder::commit
//!
//! # Quick start
//!
//! ```no_run
//! use embree3::{BufferUsage, Device, Format, GeometryKind, IntersectContext, Ray, RayHit};
//!
//! let device = Device::new().unwrap();
//! let mut scene = device.create_scene().unwrap();
//!
//! // One triangle.
//! let mut tri = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
//! tri.set_new_buffer::<[f32; 3]>(BufferUsage::VERTEX, 0, Format::FLOAT3, 3 * 4, 3)
//!     .unwrap()
//!     .copy_from_slice(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
//! tri.set_new_buffer::<[u32; 3]>(BufferUsage::INDEX, 0, Format::UINT3, 3 * 4, 1)
//!     .unwrap()
//!     .copy_from_slice(&[[0, 1, 2]]);
//! let tri = tri.commit();
//! scene.attach_geometry(&tri);
//! scene.commit();
//!
//! // Trace one ray at it.
//! let mut ctx = IntersectContext::coherent();
//! let mut rayhit = RayHit::from(Ray::segment(
//!     [0.25, 0.25, -1.0],
//!     [0.0, 0.0, 1.0],
//!     0.0,
//!     f32::INFINITY,
//! ));
//! scene.intersect(&mut ctx, &mut rayhit);
//! assert!(rayhit.hit.is_valid());
//! ```
//!
//! # Documentation
//!
//! Rust doc can be found [here](https://docs.rs/embree3/).
//! Embree documentation can be found [here](https://embree.github.io/api.html).
//! See the [examples/](https://github.com/matthiascy/embree3/tree/master/examples)
//! for some example applications using the bindings.
//!
//! # Intentionally unwrapped embree functions
//!
//! A few embree C entry points are deliberately *not* exposed, because a safer
//! or more idiomatic Rust mechanism already covers them. Reach for the
//! alternative listed here:
//!
//! | embree C function | Use instead | Why |
//! |---|---|---|
//! | `rtcNewSharedBuffer` | [`GeometryBuilder::set_shared_buffer`] | App-owned, zero-copy data is bound through a real `&'buf [u8]` borrow, so the compiler enforces "the data outlives the geometry". The C buffer *object* exists to share a raw pointer across bindings, which a Rust reference already does, so it adds no capability. |
//! | `rtcSetGeometryPointQueryFunction` | [`Scene::point_query`] | Its callback receives no per-geometry pointer (only the scene query's `userPtr`), so it cannot host a capturing closure. Branch on `geomID` inside the [`Scene::point_query`] closure for per-geometry logic. |
//! | `rtcGetSceneDevice` | [`Scene::device`] | The scene already tracks (and hands back) its [`Device`]; the raw getter would only duplicate it. |
//! | `rtcSetDeviceProperty` | *(none)* | Embree exposes **no public writable device properties** (the only settable ones are hidden internal debug integers), so a wrapper would reject every public `DeviceProperty`. Use [`Device::get_property`] for the read-only queries. |
//! | `rtcRetainBVH` | *(none)* | [`Bvh`] is a non-`Clone` build target (the build result borrows it exclusively), so there is never a second handle to retain; `rtcReleaseBVH` runs once in `Drop`. |
//! | `rtcRetainScene` | `Arc<Scene>` | [`Scene`] is a non-`Clone` unique owner of its handle (so `&mut Scene` stays exclusive for mutation/commit); share a committed scene with `Arc<Scene>`. There is never a second handle to retain, and `rtcReleaseScene` runs once in `Drop`. |
//!
//! Two further functions, `rtcGetGeometryUserData` and `rtcRetainGeometry`, are
//! not exposed because the crate's geometry ownership model (an internal `Arc`
//! plus a lock-free callback table) supersedes them; no user action is needed.
//!
//! # Panics in callbacks
//!
//! User closures registered as embree callbacks (geometry
//! intersect/occluded/bounds/filter/displacement, the scene progress monitor,
//! point queries, the BVH builder, and [`Scene::collide`]) run behind embree's
//! non-unwinding `extern "C"` ABI. A panic that escapes such a callback
//! **aborts the process** -- Rust turns an unwind that reaches a non-`-unwind`
//! `extern "C"` boundary into an abort, so this is defined behavior, not UB.
//! The hot per-ray/per-primitive trampolines therefore call the closure
//! directly (no per-call `catch_unwind` landing pad). Handle recoverable errors
//! *inside* the closure (e.g. record them in captured state); do not rely on
//! catching a panic across a query.

extern crate core;

use std::{
    alloc, mem,
    ops::{Deref, DerefMut, Not},
};

mod buffer;
mod bvh;
mod callback;
mod context;
mod device;
mod error;
mod geometry;
mod ray;
mod scene;

/// Automatically generated bindings to the Embree C API.
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(missing_docs)]
pub mod sys;

pub use buffer::*;
pub use bvh::*;
pub use context::*;
pub use device::*;
pub use geometry::*;
pub use ray::*;
pub use scene::*;

// Pull in some cleaned up enum and bitfield types directly,
// with prettier aliases

/// An axis-aligned bounding box, given by its `lower` and `upper` corners.
/// Returned by [`Scene::get_bounds`] and filled in by user-geometry bounds
/// callbacks.
pub type Bounds = sys::RTCBounds;

/// Linear (motion-blur) bounds: the axis-aligned bounding box at the start
/// (`bounds0`) and end (`bounds1`) of the scene's time range. See
/// [`Scene::get_linear_bounds`](crate::Scene::get_linear_bounds).
pub type LinearBounds = sys::RTCLinearBounds;

/// Defines the type of slots to assign data buffers to.
///
/// For most geometry types the [`BufferUsage::INDEX`] slot is used to assign
/// an index buffer, while the [`BufferUsage::VERTEX`] is used to assign the
/// corresponding vertex buffer.
///
/// The [`BufferUsage::VERTEX_ATTRIBUTE`] slot can get used to assign
/// arbitrary additional vertex data which can get interpolated using the
/// [`Geometry::interpolate`] and [`Geometry::interpolate_n`] API calls.
///
/// The [`BufferUsage::NORMAL`], [`BufferUsage::TANGENT`], and
/// [`BufferUsage::NORMAL_DERIVATIVE`] are special buffers required to assign
/// per vertex normals, tangents, and normal derivatives for some curve types.
///
/// The [`BufferUsage::GRID`] buffer is used to assign the grid primitive buffer
/// for grid geometries (see [`GeometryKind::GRID`]).
///
/// The [`BufferUsage::FACE`], [`BufferUsage::LEVEL`],
/// [`BufferUsage::EDGE_CREASE_INDEX`], [`BufferUsage::EDGE_CREASE_WEIGHT`],
/// [`BufferUsage::VERTEX_CREASE_INDEX`], [`BufferUsage::VERTEX_CREASE_WEIGHT`],
/// and [`BufferUsage::HOLE`] are special buffers required to create subdivision
/// meshes (see [`GeometryKind::SUBDIVISION`]).
///
/// [`BufferUsage::FLAGS`] can get used to add additional flag per primitive of
/// a geometry, and is currently only used for linear curves.
pub type BufferUsage = sys::RTCBufferType;

/// Speed-vs-quality trade-off for a scene or BVH build (see
/// [`Scene::set_build_quality`] and [`BuildConfig`]).
pub type BuildQuality = sys::RTCBuildQuality;

/// Flags controlling a standalone BVH build (e.g. dynamic / refittable). See
/// [`BuildConfig`].
pub type BuildFlags = sys::RTCBuildFlags;

/// Per-segment flags for curve geometries (e.g. neighbor joins).
pub type CurveFlags = sys::RTCCurveFlags;

/// A queryable, read-only device property (see [`Device::get_property`]).
pub type DeviceProperty = sys::RTCDeviceProperty;

/// An Embree error code. Returned by fallible operations and reported through
/// the device error callback; see [`Device::get_error`].
pub type Error = sys::RTCError;

/// The element format of a data buffer, e.g. [`Format::FLOAT3`] for vertex
/// positions or [`Format::UINT3`] for a triangle index.
pub type Format = sys::RTCFormat;

/// Flags on an [`IntersectContext`] selecting the traversal mode (e.g. coherent
/// vs incoherent ray distributions).
pub type IntersectContextFlags = sys::RTCIntersectContextFlags;

/// Scene-level flags (e.g. `DYNAMIC`, `ROBUST`, `COMPACT`); see
/// [`Scene::set_flags`].
pub type SceneFlags = sys::RTCSceneFlags;

/// Subdivision mode for subdivision-surface geometries (how the limit surface
/// is evaluated near boundaries and creases).
pub type SubdivisionMode = sys::RTCSubdivisionMode;
/// The type of a geometry, used to determine which geometry type to create.
pub type GeometryKind = sys::RTCGeometryType;

/// Marker trait for types usable as callback user data: geometry callback user
/// data (bound via [`GeometryBuilder`]'s `_owned` / `_borrowed` callback
/// setters) or point-query user data ([`Scene::point_query`]).
///
/// The blanket impl covers every `Send + Sync + 'static` type. The bounds are
/// required because callbacks may read the data from embree worker threads
/// (`Send + Sync`) for as long as the geometry/scene lives (`'static`).
pub trait UserData: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> UserData for T {}

/// Validity mask value for rays or hits in the filter functions.
/// See [`ValidityN`].
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValidMask {
    /// The lane is active: the ray/hit is processed. Embree's `-1`.
    Valid = -1,
    /// The lane is inactive: the ray/hit is skipped and left untouched.
    /// Embree's `0`.
    Invalid = 0,
}

impl Not for ValidMask {
    type Output = ValidMask;

    fn not(self) -> Self::Output {
        match self {
            ValidMask::Valid => ValidMask::Invalid,
            ValidMask::Invalid => ValidMask::Valid,
        }
    }
}

impl Not for &ValidMask {
    type Output = <ValidMask as Not>::Output;

    fn not(self) -> Self::Output { <ValidMask as Not>::not(*self) }
}

impl PartialEq<i32> for ValidMask {
    fn eq(&self, other: &i32) -> bool { *other == *self as i32 }
}

impl PartialEq<ValidMask> for i32 {
    fn eq(&self, other: &ValidMask) -> bool { *self == *other as i32 }
}

/// Structure that represents a quaternion decomposition of an affine
/// transformation.
///
/// The affine transformation can be decomposed into three parts:
///
/// 1. A upper triangular scaling/skew/shift matrix
///
///   ```text
///   | scale_x  skew_xy  skew_xz  shift_x |
///   |   0      scale_y  skew_yz  shift_y |
///   |   0         0     scale_z  shift_z |
///   |   0         0        0         1   |
///   ```
///
/// 2. A translation matrix
///   ```text
///   | 1   0   0 translation_x |
///   | 0   1   0 translation_y |
///   | 0   0   1 translation_z |
///   | 0   0   0       1       |
///   ```
///
/// 3. A rotation matrix R, represented as a quaternion
///   ```text
///   quaternion_r + i * quaternion_i + j * quaternion_j + k * quaternion_k
///   ```
///   where i, j, k are the imaginary unit vectors. The passed quaternion will
///   be normalized internally.
///
/// The affine transformation matrix corresponding to a quaternion decomposition
/// is TRS and a point `p = (x, y, z, 1)^T` is transformed as follows:
///
/// ```text
/// p' = T * R * S * p
/// ```
pub type QuaternionDecomposition = sys::RTCQuaternionDecomposition;

impl Default for QuaternionDecomposition {
    fn default() -> Self { QuaternionDecomposition::identity() }
}

impl QuaternionDecomposition {
    /// Create a new quaternion decomposition with the identity transformation.
    pub fn identity() -> Self {
        QuaternionDecomposition {
            scale_x: 1.0,
            scale_y: 1.0,
            scale_z: 1.0,
            skew_xy: 0.0,
            skew_xz: 0.0,
            skew_yz: 0.0,
            shift_x: 0.0,
            shift_y: 0.0,
            shift_z: 0.0,
            quaternion_r: 1.0,
            quaternion_i: 0.0,
            quaternion_j: 0.0,
            quaternion_k: 0.0,
            translation_x: 0.0,
            translation_y: 0.0,
            translation_z: 0.0,
        }
    }

    /// Returns the scale part of the decomposition.
    pub fn scale(&self) -> [f32; 3] { [self.scale_x, self.scale_y, self.scale_z] }

    /// Returns the skew part of the decomposition.
    pub fn skew(&self) -> [f32; 3] { [self.skew_xy, self.skew_xz, self.skew_yz] }

    /// Returns the shift part of the decomposition.
    pub fn shift(&self) -> [f32; 3] { [self.shift_x, self.shift_y, self.shift_z] }

    /// Returns the translation part of the decomposition.
    pub fn quaternion(&self) -> [f32; 4] {
        [
            self.quaternion_r,
            self.quaternion_i,
            self.quaternion_j,
            self.quaternion_k,
        ]
    }

    /// Set the quaternion part of the decomposition.
    pub fn set_quaternion(&mut self, quaternion: [f32; 4]) {
        self.quaternion_r = quaternion[0];
        self.quaternion_i = quaternion[1];
        self.quaternion_j = quaternion[2];
        self.quaternion_k = quaternion[3];
    }

    /// Set the scaling part of the decomposition.
    pub fn set_scale(&mut self, scale: [f32; 3]) {
        self.scale_x = scale[0];
        self.scale_y = scale[1];
        self.scale_z = scale[2];
    }

    /// Set the skew part of the decomposition.
    pub fn set_skew(&mut self, skew: [f32; 3]) {
        self.skew_xy = skew[0];
        self.skew_xz = skew[1];
        self.skew_yz = skew[2];
    }

    /// Set the shift part of the decomposition.
    pub fn set_shift(&mut self, shift: [f32; 3]) {
        self.shift_x = shift[0];
        self.shift_y = shift[1];
        self.shift_z = shift[2];
    }

    /// Set the translation part of the decomposition.
    pub fn set_translation(&mut self, translation: [f32; 3]) {
        self.translation_x = translation[0];
        self.translation_y = translation[1];
        self.translation_z = translation[2];
    }
}

/// The invalid ID for Embree intersection results (e.g. `Hit::geomID`,
/// `Hit::primID`, etc.)
pub const INVALID_ID: u32 = u32::MAX;

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            lower_x: f32::INFINITY,
            lower_y: f32::INFINITY,
            lower_z: f32::INFINITY,
            align0: 0.0,
            upper_x: f32::INFINITY,
            upper_y: f32::INFINITY,
            upper_z: f32::INFINITY,
            align1: 0.0,
        }
    }
}

impl Bounds {
    /// Returns the lower bounds of the bounding box.
    pub fn lower(&self) -> [f32; 3] { [self.lower_x, self.lower_y, self.lower_z] }

    /// Returns the upper bounds of the bounding box.
    pub fn upper(&self) -> [f32; 3] { [self.upper_x, self.upper_y, self.upper_z] }
}

/// Object used to traverses the BVH and calls a user defined callback function
/// for each primitive of the scene that intersects the query domain.
///
/// See [`Scene::point_query`] for more information.
pub type PointQuery = sys::RTCPointQuery;

/// A SoA packet of 4 point queries (see [`Scene::point_query4`]).
pub type PointQuery4 = sys::RTCPointQuery4;
/// A SoA packet of 8 point queries (see [`Scene::point_query8`]).
pub type PointQuery8 = sys::RTCPointQuery8;
/// A SoA packet of 16 point queries (see [`Scene::point_query16`]).
pub type PointQuery16 = sys::RTCPointQuery16;

/// Primitives that can be used to build a BVH.
pub type BuildPrimitive = sys::RTCBuildPrimitive;

/// A candidate colliding primitive pair reported by [`Scene::collide`]: the
/// `geomID`/`primID` of one primitive in each scene. It is a
/// potentially-intersecting pair from a leaf pair reached during broad-phase
/// traversal; embree does not test the bounds before reporting, so the pair's
/// bounds need not overlap and the callback must narrow-phase.
pub type Collision = sys::RTCCollision;

/// Utility for making specifically aligned vector.
///
/// This is a growable, dynamically allocated, arbitrarily aligned container.
/// Please use [`AlignedArray`] if you only need a 16 bytes aligned, fix-sized
/// storage.
///
/// This is a wrapper around `Vec` that ensures the alignment of the vector.
/// The reason for this is that memory must be deallocated with the
/// same alignment as it was allocated with. This is not guaranteed if
/// we allocate a memory block with the alignment then cast it to a
/// `Vec` of `T` and then drop it, since the `Vec` will deallocate the
/// memory with the alignment of `T`.
pub struct AlignedVector<T> {
    vec: Vec<T>,
    layout: alloc::Layout,
}

impl<T> AlignedVector<T> {
    /// Allocate `len` zeroed elements, aligned to at least `align` bytes (and
    /// at least `T`'s own alignment).
    pub fn zeroed(len: usize, align: usize) -> Self {
        let t_size = mem::size_of::<T>();
        let t_align = mem::align_of::<T>();
        let align = t_align.max(align);
        let layout = alloc::Layout::from_size_align(t_size * len, align).unwrap();
        unsafe {
            let raw = alloc::alloc_zeroed(layout);
            if raw.is_null() {
                alloc::handle_alloc_error(layout);
            }
            AlignedVector {
                vec: Vec::from_raw_parts(raw as *mut T, len, len),
                layout,
            }
        }
    }

    /// Allocate `len` elements aligned to at least `align` bytes, each
    /// initialized to `init`.
    pub fn new_init(len: usize, align: usize, init: T) -> Self
    where
        T: Copy,
    {
        let mut v = Self::zeroed(len, align);
        for x in v.iter_mut() {
            *x = init;
        }
        v
    }

    /// Returns the alignment of the vector.
    pub fn alignment(&self) -> usize { self.layout.align() }
}

impl<T> Deref for AlignedVector<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target { &self.vec }
}

impl<T> DerefMut for AlignedVector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.vec }
}

impl<T> Drop for AlignedVector<T> {
    fn drop(&mut self) {
        unsafe {
            let mut vec = mem::take(&mut self.vec);
            let raw = vec.as_mut_ptr() as *mut u8;
            alloc::dealloc(raw, self.layout);
            mem::forget(vec);
        }
    }
}

#[test]
fn test_aligned_vector_alloc() {
    let v = AlignedVector::<f32>::new_init(24, 16, 1.0);
    for x in v.iter() {
        assert_eq!(*x, 1.0);
    }
}

#[test]
fn miri_aligned_vector_new_is_initialised() {
    // `new` claims len elements are initialised over memory from `alloc::alloc`,
    // which is undefined. Under Miri this reads uninitialised memory.
    let v = AlignedVector::<u32>::zeroed(8, 16);
    let mut acc = 0u32;
    for x in v.iter() {
        acc = acc.wrapping_add(*x); // reading uninitialised T -> Miri error
    }
    std::hint::black_box(acc);
    assert_eq!(v.len(), 8);
}

/// 16 bytes aligned with known size at compile time.
#[repr(align(16))]
pub struct AlignedArray<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> Deref for AlignedArray<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T, const N: usize> DerefMut for AlignedArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

/// Utility function to normalise a vector.
#[inline(always)]
fn normalise_vector3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let len_inv = if len_sq.is_finite() && len_sq != 0.0 {
        len_sq.sqrt().recip()
    } else {
        0.0
    };

    [v[0] * len_inv, v[1] * len_inv, v[2] * len_inv]
}

#[test]
fn test_normalise_vector3() {
    let v = normalise_vector3([1.0, 2.0, 3.0]);
    assert_eq!(v[0], 0.26726124);
    assert_eq!(v[1], 0.5345225);
    assert_eq!(v[2], 0.8017837);

    let v = normalise_vector3([0.0, 0.0, 0.0]);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 0.0);

    let v = normalise_vector3([1.0, 0.0, 0.0]);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 0.0);
}
