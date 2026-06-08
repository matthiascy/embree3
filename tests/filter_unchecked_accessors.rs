//! End-to-end exercise of the public `unsafe` unchecked SoA accessors from
//! inside a real intersect-filter callback: `ValidityN::get_unchecked` /
//! `set_unchecked`, `RayN::gather_unchecked`, and `HitN::gather_unchecked`.
//!
//! The pure-Rust unit tests in `src/` prove the unchecked column offsets agree
//! with the checked accessors; this proves the path *works through embree* --
//! the filter reads each active lane with the unchecked gathers (recording what
//! it saw) and rejects the lane with the unchecked validity setter, and the
//! final ray result reflects that. (`RayN::set_tfar_unchecked` /
//! `HitN::scatter_unchecked` are exercised end-to-end by the lane-handle path
//! in `tests/lane_iter.rs`.)
mod common;

use std::sync::{Arc, Mutex};

use embree3::{HitN, IntersectContext, Ray, RayHit, RayN, ValidityN};

#[derive(Clone, Copy, Default)]
struct Seen {
    ran: bool,
    valid0: i32,
    geom_id: u32,
    org: [f32; 3],
}

#[test]
fn filter_uses_unchecked_accessors_end_to_end() {
    let device = common::device();
    let mut tri = common::unit_triangle(&device);

    let seen = Arc::new(Mutex::new(Seen::default()));
    let sink = seen.clone();
    tri.set_intersect_filter_function::<_, ()>(
        move |rays: RayN<'_>,
              hits: HitN<'_>,
              mut valid: ValidityN<'_>,
              _ctx: &mut IntersectContext,
              _user: Option<&()>| {
            for i in 0..rays.len() {
                // Read validity through the unchecked accessor (the lane index is
                // proven in range by `0..rays.len()`).
                let v = unsafe { valid.get_unchecked(i) };
                if v == 0 {
                    continue;
                }
                // Gather the ray and the candidate hit through the unchecked accessors.
                let ray = unsafe { rays.gather_unchecked(i) };
                let hit = unsafe { hits.gather_unchecked(i) };
                {
                    let mut s = sink.lock().unwrap();
                    s.ran = true;
                    s.valid0 = v;
                    s.geom_id = hit.geomID;
                    s.org = [ray.org_x, ray.org_y, ray.org_z];
                }
                // Reject this lane through the unchecked setter.
                unsafe { valid.set_unchecked(i, 0) };
            }
        },
    );

    let tri = tri.commit();
    let mut scene = device.create_scene().unwrap();
    scene.attach_geometry(&tri);
    scene.commit();

    let mut ctx = IntersectContext::coherent();
    let mut rh = RayHit::from(Ray::segment(
        [0.25, 0.25, -1.0],
        [0.0, 0.0, 1.0],
        0.0,
        f32::INFINITY,
    ));
    scene.intersect(&mut ctx, &mut rh);

    let s = *seen.lock().unwrap();
    assert!(
        s.ran,
        "the filter must have run via the unchecked accessors"
    );
    assert_eq!(s.valid0, -1, "active lane reads -1 through get_unchecked");
    assert_eq!(s.geom_id, 0, "gather_unchecked read the triangle's geomID");
    assert_eq!(
        s.org,
        [0.25, 0.25, -1.0],
        "gather_unchecked read the ray's origin"
    );
    assert!(
        !rh.hit.is_valid(),
        "set_unchecked(0) rejected the lane, so the final result is a miss"
    );
}
