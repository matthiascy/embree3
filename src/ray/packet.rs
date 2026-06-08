//! Ray packet types and traits.

use crate::{
    normalise_vector3, sys, Hit, Ray, RayHit, SoAHit, SoAHitIter, SoAHitRef, SoARay, SoARayIter,
    SoARayIterMut, INVALID_ID,
};
use std::marker::PhantomData;

/// A ray packet of size 4.
pub type Ray4 = sys::RTCRay4;

/// A hit packet of size 4.
pub type Hit4 = sys::RTCHit4;

/// A ray/hit packet of size 4.
pub type RayHit4 = sys::RTCRayHit4;

/// A ray packet of size 8.
pub type Ray8 = sys::RTCRay8;

/// A hit packet of size 8.
pub type Hit8 = sys::RTCHit8;

/// A ray/hit packet of size 8.
pub type RayHit8 = sys::RTCRayHit8;

/// A ray packet of size 16.
pub type Ray16 = sys::RTCRay16;

/// A hit packet of size 16.
pub type Hit16 = sys::RTCHit16;

/// A ray/hit packet of size 16.
pub type RayHit16 = sys::RTCRayHit16;

/// Represents a packet of rays.
///
/// Used as a trait bound for functions that operate on ray packets.
/// See [`occluded_stream_aos`](`crate::Scene::occluded_stream_aos`) and
/// [`intersect_stream_aos`](`crate::Scene::intersect_stream_aos`).
pub trait RayPacket: Sized {
    const LEN: usize;
}

/// Represents a packet of hits.
pub trait HitPacket: Sized {
    const LEN: usize;
}

/// Represents a packet of ray/hit pairs.
pub trait RayHitPacket: Sized {
    type Ray: RayPacket;
    type Hit: HitPacket;
    const LEN: usize = Self::Ray::LEN;
}

macro_rules! impl_packet_traits {
    ($($ray:ident, $hit:ident, $rayhit:ident, $n:expr);*) => {
        $(
            impl RayPacket for $ray {
                const LEN: usize = $n;
            }

            impl HitPacket for $hit {
                const LEN: usize = $n;
            }

            impl RayHitPacket for $rayhit {
                type Ray = $ray;
                type Hit = $hit;
            }
        )*
    }
}

impl_packet_traits! {
    Ray, Hit, RayHit, 1;
    Ray4, Hit4, RayHit4, 4;
    Ray8, Hit8, RayHit8, 8;
    Ray16, Hit16, RayHit16, 16
}

macro_rules! impl_ray_packets {
    ($($t:ident, $n:expr);*) => {
        $(
            impl $t {
                pub const fn new(origin: [[f32; 3]; $n], dir: [[f32; 3]; $n]) -> $t {
                    $t::segment(origin, dir, [0.0; $n], [f32::INFINITY; $n])
                }

                pub const fn segment(origin: [[f32; 3]; $n], dir: [[f32; 3]; $n], tnear: [f32; $n], tfar: [f32; $n]) -> $t {
                    let [org_x, org_y, org_z, dir_x, dir_y, dir_z] = {
                        let mut elems = [[0.0f32; $n]; 6];
                        let mut i = 0;
                        while i < $n {
                            elems[0][i] = origin[i][0];
                            elems[1][i] = origin[i][1];
                            elems[2][i] = origin[i][2];
                            elems[3][i] = dir[i][0];
                            elems[4][i] = dir[i][1];
                            elems[5][i] = dir[i][2];
                            i += 1;
                        }
                        elems
                    };
                    Self {
                        org_x,
                        org_y,
                        org_z,
                        dir_x,
                        dir_y,
                        dir_z,
                        tnear,
                        tfar,
                        time: [0.0; $n],
                        mask: [u32::MAX; $n],
                        id: [0; $n],
                        flags: [0; $n],
                    }
                }

                pub const fn empty() -> $t {
                    $t::segment(
                        [[0.0, 0.0, 0.0]; $n],
                        [[0.0, 0.0, 0.0]; $n],
                        [0.0; $n],
                        [f32::INFINITY; $n],
                    )
                }

                pub fn iter(&self) -> SoARayIter<'_, $t> { SoARayIter::new(self, $n) }

                pub fn iter_mut(&mut self) -> SoARayIterMut<'_, $t> { SoARayIterMut::new(self, $n) }
            }

            impl Default for $t {
                fn default() -> Self { Self::empty() }
            }

            impl SoARay for $t {
                #[inline]
                fn org(&self, i: usize) -> [f32; 3] { [self.org_x[i], self.org_y[i], self.org_z[i]] }

                #[inline]
                fn set_org(&mut self, i: usize, o: [f32; 3]) {
                    self.org_x[i] = o[0];
                    self.org_y[i] = o[1];
                    self.org_z[i] = o[2];
                }

                #[inline]
                fn tnear(&self, i: usize) -> f32 { self.tnear[i] }

                #[inline]
                fn set_tnear(&mut self, i: usize, t: f32) { self.tnear[i] = t }

                #[inline]
                fn dir(&self, i: usize) -> [f32; 3] { [self.dir_x[i], self.dir_y[i], self.dir_z[i]] }

                #[inline]
                fn set_dir(&mut self, i: usize, d: [f32; 3]) {
                    self.dir_x[i] = d[0];
                    self.dir_y[i] = d[1];
                    self.dir_z[i] = d[2];
                }

                #[inline]
                fn time(&self, i: usize) -> f32 { self.time[i] }

                #[inline]
                fn set_time(&mut self, i: usize, t: f32) { self.time[i] = t }

                #[inline]
                fn tfar(&self, i: usize) -> f32 { self.tfar[i] }

                #[inline]
                fn set_tfar(&mut self, i: usize, t: f32) { self.tfar[i] = t}

                #[inline]
                fn mask(&self, i: usize) -> u32 { self.mask[i] }

                #[inline]
                fn set_mask(&mut self, i: usize, m: u32) { self.mask[i] = m }

                #[inline]
                fn id(&self, i: usize) -> u32 { self.id[i] }

                #[inline]
                fn set_id(&mut self, i: usize, id: u32) { self.id[i] = id }

                #[inline]
                fn flags(&self, i: usize) -> u32 { self.flags[i] }

                #[inline]
                fn set_flags(&mut self, i: usize, f: u32) { self.flags[i] = f }
            }
        )*
    };
}

impl_ray_packets!(Ray4, 4; Ray8, 8; Ray16, 16);

macro_rules! impl_hit_packets {
    ($($t:ident, $n:expr);*) => {
        $(
            impl $t {
                pub fn new() -> $t {
                    $t {
                        Ng_x: [0.0; $n],
                        Ng_y: [0.0; $n],
                        Ng_z: [0.0; $n],
                        u: [0.0; $n],
                        v: [0.0; $n],
                        primID: [INVALID_ID; $n],
                        geomID: [INVALID_ID; $n],
                        instID: [[INVALID_ID; $n]],
                    }
                }
                pub fn any_hit(&self) -> bool { self.iter_validity().any(|h| h) }

                pub fn iter_validity(&self) -> impl Iterator<Item = bool> + '_ {
                    self.geomID.iter().map(|g| *g != INVALID_ID)
                }

                pub fn iter(&self) -> SoAHitIter<'_, $t> { SoAHitIter::new(self, $n) }

                pub fn iter_hits(&self) -> impl Iterator<Item = SoAHitRef<'_, $t>> {
                    SoAHitIter::new(self, 4).filter(|h| h.is_valid())
                }
            }

            impl Default for $t {
                fn default() -> Self { Self::new() }
            }

            impl SoAHit for $t {
                #[inline]
                fn normal(&self, i: usize) -> [f32; 3] { [self.Ng_x[i], self.Ng_y[i], self.Ng_z[i]] }

                #[inline]
                fn unit_normal(&self, i: usize) -> [f32; 3] {
                    let n = self.normal(i);
                    let len = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                    if len > 0.0 {
                        let inv_len = 1.0 / len.sqrt();
                        [n[0] * inv_len, n[1] * inv_len, n[2] * inv_len]
                    } else {
                        [0.0, 0.0, 0.0]
                    }
                }

                #[inline]
                fn set_normal(&mut self, i: usize, n: [f32; 3]) {
                    self.Ng_x[i] = n[0];
                    self.Ng_y[i] = n[1];
                    self.Ng_z[i] = n[2];
                }

                #[inline]
                fn u(&self, i: usize) -> f32 { self.u[i] }

                #[inline]
                fn v(&self, i: usize) -> f32 { self.v[i] }

                #[inline]
                fn uv(&self, i: usize) -> [f32; 2] { [self.u[i], self.v[i]] }

                #[inline]
                fn set_u(&mut self, i: usize, u: f32) { self.u[i] = u; }

                #[inline]
                fn set_v(&mut self, i: usize, v: f32) { self.v[i] = v; }

                #[inline]
                fn set_uv(&mut self, i: usize, uv: [f32; 2]) {
                    self.u[i] = uv[0];
                    self.v[i] = uv[1];
                }

                #[inline]
                fn prim_id(&self, i: usize) -> u32 { self.primID[i] }

                #[inline]
                fn set_prim_id(&mut self, i: usize, id: u32) { self.primID[i] = id; }

                #[inline]
                fn geom_id(&self, i: usize) -> u32 { self.geomID[i] }

                #[inline]
                fn set_geom_id(&mut self, i: usize, id: u32) { self.geomID[i] = id; }

                #[inline]
                fn inst_id(&self, i: usize) -> u32 { self.instID[0][i] }

                #[inline]
                fn set_inst_id(&mut self, i: usize, id: u32) { self.instID[0][i] = id; }
            }
        )*
    };
}

impl_hit_packets!(Hit4, 4; Hit8, 8; Hit16, 16);

impl RayHit4 {
    pub fn new(ray: Ray4) -> RayHit4 {
        sys::RTCRayHit4 {
            ray,
            hit: Hit4::new(),
        }
    }
    pub fn iter(&self) -> std::iter::Zip<SoARayIter<'_, Ray4>, SoAHitIter<'_, Hit4>> {
        self.ray.iter().zip(self.hit.iter())
    }
}

/// Ray packet of runtime size.
///
/// It is used to represent a packet of rays that is not known at compile
/// time, generally used as an argument to callback functions. The size
/// of the packet can only be either 1, 4, 8, or 16.
///
/// For ray streams, use [`RayNp`](`crate::ray::RayNp`).
pub struct RayN<'a> {
    pub(crate) ptr: *mut sys::RTCRayN,
    pub(crate) len: usize,
    pub(crate) marker: PhantomData<&'a mut sys::RTCRayN>,
}

impl<'a> RayN<'a> {
    /// Returns the number of rays in the packet.
    ///
    /// Can be either 1, 4, 8, or 16.
    pub fn len(&self) -> usize { self.len }

    /// Returns true if the packet is empty.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Returns the hit point of the `i`th ray.
    pub fn hit_point(&self, i: usize) -> [f32; 3] {
        assert!(i < self.len, "index out of bounds");
        let mut p = self.org(i);
        let d = self.dir(i);
        let t = self.tfar(i);
        p[0] += d[0] * t;
        p[1] += d[1] * t;
        p[2] += d[2] * t;
        p
    }
}

impl<'a> SoARay for RayN<'a> {
    #[inline]
    fn org(&self, i: usize) -> [f32; 3] {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            let ptr = self.ptr as *const f32;
            [
                *ptr.add(i),
                *ptr.add(self.len + i),
                *ptr.add(2 * self.len + i),
            ]
        }
    }

    #[inline]
    fn set_org(&mut self, i: usize, o: [f32; 3]) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            let ptr = self.ptr as *mut f32;
            *ptr.add(i) = o[0];
            *ptr.add(self.len + i) = o[1];
            *ptr.add(2 * self.len + i) = o[2];
        }
    }

    #[inline]
    fn dir(&self, i: usize) -> [f32; 3] {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            let ptr = self.ptr as *const f32;
            [
                *ptr.add(4 * self.len + i),
                *ptr.add(5 * self.len + i),
                *ptr.add(6 * self.len + i),
            ]
        }
    }

    #[inline]
    fn set_dir(&mut self, i: usize, d: [f32; 3]) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            let ptr = self.ptr as *mut f32;
            *ptr.add(4 * self.len + i) = d[0];
            *ptr.add(5 * self.len + i) = d[1];
            *ptr.add(6 * self.len + i) = d[2];
        }
    }

    #[inline]
    fn tnear(&self, i: usize) -> f32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const f32).add(3 * self.len + i) }
    }

    #[inline]
    fn set_tnear(&mut self, i: usize, t: f32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut f32).add(3 * self.len + i) = t;
        }
    }

    #[inline]
    fn tfar(&self, i: usize) -> f32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const f32).add(8 * self.len + i) }
    }

    #[inline]
    fn set_tfar(&mut self, i: usize, t: f32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut f32).add(8 * self.len + i) = t;
        }
    }

    #[inline]
    fn time(&self, i: usize) -> f32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const f32).add(7 * self.len + i) }
    }

    #[inline]
    fn set_time(&mut self, i: usize, t: f32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut f32).add(7 * self.len + i) = t;
        }
    }

    #[inline]
    fn mask(&self, i: usize) -> u32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const u32).add(9 * self.len + i) }
    }

    #[inline]
    fn set_mask(&mut self, i: usize, m: u32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut u32).add(9 * self.len + i) = m;
        }
    }

    #[inline]
    fn id(&self, i: usize) -> u32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const u32).add(10 * self.len + i) }
    }

    #[inline]
    fn set_id(&mut self, i: usize, id: u32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut u32).add(10 * self.len + i) = id;
        }
    }

    #[inline]
    fn flags(&self, i: usize) -> u32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const u32).add(11 * self.len + i) }
    }

    #[inline]
    fn set_flags(&mut self, i: usize, f: u32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut u32).add(11 * self.len + i) = f;
        }
    }
}

impl<'a> RayN<'a> {
    /// Gather lane `i` into a single-ray [`Ray`] **with no bounds check**.
    ///
    /// The checked [`SoARay`] accessors bounds-check every field access
    /// unconditionally (they `assert!`, even in release). When a filter
    /// callback iterates the packet over a lane index it has already proven
    /// in range (e.g. `for i in 0..rays.len()`), that per-field check is
    /// redundant; this reads the whole ray in one shot with no check.
    /// `#[inline(always)]` so the SoA reads inline into the caller with no
    /// out-of-line call or per-field `cmp`/panic branch. The column offsets
    /// match the checked accessors above (a unit test asserts they agree).
    ///
    /// # Safety
    ///
    /// `i < self.len()`.
    #[inline(always)]
    pub unsafe fn gather_unchecked(&self, i: usize) -> Ray {
        let n = self.len;
        let f = self.ptr as *const f32;
        let u = self.ptr as *const u32;
        Ray {
            org_x: *f.add(i),
            org_y: *f.add(n + i),
            org_z: *f.add(2 * n + i),
            tnear: *f.add(3 * n + i),
            dir_x: *f.add(4 * n + i),
            dir_y: *f.add(5 * n + i),
            dir_z: *f.add(6 * n + i),
            time: *f.add(7 * n + i),
            tfar: *f.add(8 * n + i),
            mask: *u.add(9 * n + i),
            id: *u.add(10 * n + i),
            flags: *u.add(11 * n + i),
        }
    }

    /// Scatter `tfar` into lane `i` **with no bounds check** (the unchecked
    /// counterpart of [`SoARay::set_tfar`]).
    ///
    /// # Safety
    ///
    /// `i < self.len()`.
    #[inline(always)]
    pub unsafe fn set_tfar_unchecked(&mut self, i: usize, tfar: f32) {
        *(self.ptr as *mut f32).add(8 * self.len + i) = tfar;
    }
}

/// Hit packet of runtime size.
///
/// It is used to represent a packet of hits that is not known at compile
/// time, generally used as an argument to callback functions. The size
/// of the packet can only be either 1, 4, 8, or 16.
pub struct HitN<'a> {
    pub(crate) ptr: *mut sys::RTCHitN,
    pub(crate) len: usize,
    pub(crate) marker: PhantomData<&'a mut sys::RTCHitN>,
}

impl<'a> SoAHit for HitN<'a> {
    #[inline]
    fn normal(&self, i: usize) -> [f32; 3] {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            [
                *(self.ptr as *const f32).add(i),
                *(self.ptr as *const f32).add(self.len + i),
                *(self.ptr as *const f32).add(2 * self.len + i),
            ]
        }
    }

    #[inline]
    fn unit_normal(&self, i: usize) -> [f32; 3] { normalise_vector3(self.normal(i)) }

    #[inline]
    fn set_normal(&mut self, i: usize, n: [f32; 3]) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            let ptr = self.ptr as *mut f32;
            *(ptr).add(i) = n[0];
            *(ptr).add(self.len + i) = n[1];
            *(ptr).add(2 * self.len + i) = n[2];
        }
    }

    #[inline]
    fn uv(&self, i: usize) -> [f32; 2] {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            [
                *(self.ptr as *const f32).add(3 * self.len + i),
                *(self.ptr as *const f32).add(4 * self.len + i),
            ]
        }
    }

    #[inline]
    fn u(&self, i: usize) -> f32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const f32).add(3 * self.len + i) }
    }

    #[inline]
    fn v(&self, i: usize) -> f32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const f32).add(4 * self.len + i) }
    }

    #[inline]
    fn set_uv(&mut self, i: usize, uv: [f32; 2]) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            let ptr = self.ptr as *mut f32;
            *(ptr).add(3 * self.len + i) = uv[0];
            *(ptr).add(4 * self.len + i) = uv[1];
        }
    }

    #[inline]
    fn set_u(&mut self, i: usize, u: f32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut f32).add(3 * self.len + i) = u;
        }
    }

    #[inline]
    fn set_v(&mut self, i: usize, v: f32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut f32).add(4 * self.len + i) = v;
        }
    }

    #[inline]
    fn prim_id(&self, i: usize) -> u32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const u32).add(5 * self.len + i) }
    }

    #[inline]
    fn set_prim_id(&mut self, i: usize, id: u32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut u32).add(5 * self.len + i) = id;
        }
    }

    #[inline]
    fn geom_id(&self, i: usize) -> u32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const u32).add(6 * self.len + i) }
    }

    #[inline]
    fn set_geom_id(&mut self, i: usize, id: u32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut u32).add(6 * self.len + i) = id;
        }
    }

    #[inline]
    fn inst_id(&self, i: usize) -> u32 {
        assert!(i < self.len, "index out of bounds");
        unsafe { *(self.ptr as *const u32).add(7 * self.len + i) }
    }

    #[inline]
    fn set_inst_id(&mut self, i: usize, id: u32) {
        assert!(i < self.len, "index out of bounds");
        unsafe {
            *(self.ptr as *mut u32).add(7 * self.len + i) = id;
        }
    }
}

impl<'a> HitN<'a> {
    /// Returns the number of hits in the packet.
    pub const fn len(&self) -> usize { self.len }

    /// Returns true if the packet is empty.
    pub const fn is_empty(&self) -> bool { self.len == 0 }

    /// Gather lane `i`'s candidate hit into a single [`Hit`] **with no bounds
    /// check**.
    ///
    /// The unchecked-read counterpart of the [`SoAHit`] accessors, for a filter
    /// callback iterating over an already-proven-in-range lane index: it reads
    /// the whole hit in one shot, skipping the per-field `assert!` the checked
    /// accessors do in release. Column offsets match the checked accessors (a
    /// unit test asserts they agree).
    ///
    /// # Safety
    ///
    /// `i < self.len()`.
    #[inline(always)]
    pub unsafe fn gather_unchecked(&self, i: usize) -> Hit {
        let n = self.len;
        let f = self.ptr as *const f32;
        let u = self.ptr as *const u32;
        Hit {
            Ng_x: *f.add(i),
            Ng_y: *f.add(n + i),
            Ng_z: *f.add(2 * n + i),
            u: *f.add(3 * n + i),
            v: *f.add(4 * n + i),
            primID: *u.add(5 * n + i),
            geomID: *u.add(6 * n + i),
            instID: [*u.add(7 * n + i)],
        }
    }

    /// Scatter a single [`Hit`] into lane `i` **with no bounds check**.
    ///
    /// `#[inline(always)]` so the SoA writes inline into the caller (a
    /// proven-in-range lane handle). Column offsets match the checked
    /// [`SoAHit`] setters above (a unit test asserts they agree).
    ///
    /// # Safety
    ///
    /// `i < self.len()`.
    #[inline(always)]
    pub unsafe fn scatter_unchecked(&mut self, i: usize, hit: &Hit) {
        let n = self.len;
        let f = self.ptr as *mut f32;
        let u = self.ptr as *mut u32;
        *f.add(i) = hit.Ng_x;
        *f.add(n + i) = hit.Ng_y;
        *f.add(2 * n + i) = hit.Ng_z;
        *f.add(3 * n + i) = hit.u;
        *f.add(4 * n + i) = hit.v;
        *u.add(5 * n + i) = hit.primID;
        *u.add(6 * n + i) = hit.geomID;
        *u.add(7 * n + i) = hit.instID[0];
    }
}

/// Combined ray and hit packet of runtime size.
///
/// The size of the packet can only be either 1, 4, 8, or 16.
pub struct RayHitN<'a> {
    pub(crate) ptr: *mut sys::RTCRayHitN,
    pub(crate) len: usize,
    pub(crate) marker: PhantomData<&'a mut sys::RTCRayHitN>,
}

impl<'a> RayHitN<'a> {
    /// Returns the ray packet.
    pub fn ray_n(&'a self) -> RayN<'a> {
        RayN {
            ptr: self.ptr as *mut sys::RTCRayN,
            len: self.len,
            marker: PhantomData,
        }
    }

    /// Returns the hit packet.
    pub fn hit_n(&'a self) -> HitN<'a> {
        HitN {
            ptr: unsafe { (self.ptr as *const u32).add(12 * self.len) as *mut sys::RTCHitN },
            len: self.len,
            marker: PhantomData,
        }
    }

    /// Returns the number of ray/hit pairs in the packet.
    pub fn len(&self) -> usize { self.len }

    /// Returns true if the packet is empty.
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

#[cfg(test)]
mod oob_tests {
    //! Lane bounds checks must be unconditional (not `debug_assert!`), so an
    //! out-of-bounds lane panics in release as well as debug. Pure-Rust (no
    //! FFI): a `RayN` view over a stack SoA buffer of `len` lanes.
    use super::*;

    fn ray_n_over(buf: &mut [f32], len: usize) -> RayN<'_> {
        RayN {
            ptr: buf.as_mut_ptr() as *mut sys::RTCRayN,
            len,
            marker: PhantomData,
        }
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn ray_n_org_out_of_bounds_panics() {
        let mut buf = [0.0f32; 12 * 4]; // RTCRayN SoA: at least 12 floats per lane
        let r = ray_n_over(&mut buf, 4);
        let _ = r.org(4); // lane 4 is outside [0, 4)
    }

    #[test]
    fn ray_n_org_in_bounds_ok() {
        let mut buf = [0.0f32; 12 * 4];
        let r = ray_n_over(&mut buf, 4);
        assert_eq!(r.org(3), [0.0, 0.0, 0.0]);
    }

    // Guards the unchecked gather/scatter column offsets against the checked
    // `SoARay`/`SoAHit` accessors: if either drifts, these fail.
    #[test]
    fn gather_unchecked_matches_checked_ray() {
        let mut r4 = Ray4::new(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
        );
        for i in 0..4 {
            r4.set_tfar(i, 100.0 + i as f32);
            r4.set_time(i, 0.25 * i as f32);
            r4.set_mask(i, i as u32 + 1);
            r4.set_id(i, i as u32 * 7);
            r4.set_flags(i, i as u32 + 3);
        }
        let view = RayN {
            ptr: &mut r4 as *mut Ray4 as *mut sys::RTCRayN,
            len: 4,
            marker: PhantomData,
        };
        for i in 0..4 {
            let u = unsafe { view.gather_unchecked(i) };
            assert_eq!([u.org_x, u.org_y, u.org_z], view.org(i));
            assert_eq!([u.dir_x, u.dir_y, u.dir_z], view.dir(i));
            assert_eq!(u.tnear, view.tnear(i));
            assert_eq!(u.tfar, view.tfar(i));
            assert_eq!(u.time, view.time(i));
            assert_eq!(u.mask, view.mask(i));
            assert_eq!(u.id, view.id(i));
            assert_eq!(u.flags, view.flags(i));
        }
    }

    #[test]
    fn scatter_unchecked_matches_checked_hit() {
        let mut h4 = Hit4::new();
        let mut view = HitN {
            ptr: &mut h4 as *mut Hit4 as *mut sys::RTCHitN,
            len: 4,
            marker: PhantomData,
        };
        let hit = Hit {
            Ng_x: 1.0,
            Ng_y: 2.0,
            Ng_z: 3.0,
            u: 0.5,
            v: 0.6,
            primID: 7,
            geomID: 9,
            instID: [11],
        };
        unsafe { view.scatter_unchecked(2, &hit) };
        assert_eq!(view.normal(2), [1.0, 2.0, 3.0]);
        assert_eq!(view.uv(2), [0.5, 0.6]);
        assert_eq!(view.prim_id(2), 7);
        assert_eq!(view.geom_id(2), 9);
        assert_eq!(view.inst_id(2), 11);
    }

    #[test]
    fn gather_unchecked_matches_checked_hit() {
        // Round-trips a known hit through the checked setters, then reads it back
        // with the unchecked gather: the column offsets of `HitN::gather_unchecked`
        // must agree with the checked `SoAHit` accessors.
        let mut h4 = Hit4::new();
        let mut view = HitN {
            ptr: &mut h4 as *mut Hit4 as *mut sys::RTCHitN,
            len: 4,
            marker: PhantomData,
        };
        let hit = Hit {
            Ng_x: 1.0,
            Ng_y: 2.0,
            Ng_z: 3.0,
            u: 0.5,
            v: 0.6,
            primID: 7,
            geomID: 9,
            instID: [11],
        };
        view.set_normal(1, [hit.Ng_x, hit.Ng_y, hit.Ng_z]);
        view.set_uv(1, [hit.u, hit.v]);
        view.set_prim_id(1, hit.primID);
        view.set_geom_id(1, hit.geomID);
        view.set_inst_id(1, hit.instID[0]);

        let g = unsafe { view.gather_unchecked(1) };
        assert_eq!([g.Ng_x, g.Ng_y, g.Ng_z], view.normal(1));
        assert_eq!([g.u, g.v], view.uv(1));
        assert_eq!(g.primID, view.prim_id(1));
        assert_eq!(g.geomID, view.geom_id(1));
        assert_eq!(g.instID[0], view.inst_id(1));
    }
}
