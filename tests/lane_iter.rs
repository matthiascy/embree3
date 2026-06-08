//! `for_each_active_lane` drives a user-geometry intersect callback through
//! the `IntersectLane` handle (`ray`/`prim_id`/`geom_id`/`filter_intersection`/
//! `commit_hit`) without threading the lane index by hand. Proves the ergonomic
//! lane iterator reports a hit end-to-end.
mod common;

use embree3::{Hit, IntersectContext, IntersectFunctionNArgs, Ray, RayHit, INVALID_ID};

#[test]
fn for_each_active_lane_reports_a_hit() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    // USER geometry with a unit-box bound at the origin (set by `user_sphere`).
    let mut geom = common::user_sphere(&device);

    geom.set_intersect_function::<_, ()>(move |args: &mut IntersectFunctionNArgs<'_, ()>| {
        args.for_each_active_lane(|mut lane| {
            let mut ray = lane.ray();
            let t = 1.5_f32; // box front face for a +z ray from z = -2
            if t > ray.tnear && t < ray.tfar {
                let mut hit = Hit {
                    Ng_x: 0.0,
                    Ng_y: 0.0,
                    Ng_z: -1.0,
                    u: 0.0,
                    v: 0.0,
                    primID: lane.prim_id(),
                    geomID: lane.geom_id(),
                    instID: [INVALID_ID],
                };
                ray.tfar = t;
                if lane.filter_intersection(&mut ray, &mut hit) {
                    lane.commit_hit(&ray, &hit);
                }
            }
        });
    });

    let geom = geom.commit();
    scene.attach_geometry(&geom);
    scene.commit();

    let mut ctx = IntersectContext::coherent();
    let mut rh = RayHit::from(Ray::segment(
        [0.0, 0.0, -2.0],
        [0.0, 0.0, 1.0],
        0.0,
        f32::INFINITY,
    ));
    scene.intersect(&mut ctx, &mut rh);

    assert!(
        rh.hit.is_valid(),
        "for_each_active_lane should have committed the hit"
    );
    assert_eq!(rh.hit.geomID, 0, "hit came from the attached user geometry");
}
