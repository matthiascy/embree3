//! Proves a user-geometry intersect callback can invoke the geometry's
//! intersection filter chain via `filter_intersection`, that the filter runs
//! with **live captured state** (no use-after-free), and that the filter's
//! accept/reject decision is honoured.
mod common;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use embree3::{
    Hit, IntersectContext, IntersectFunctionNArgs, OccludedFunctionNArgs, Ray4, RayHit4, SoAHit,
    SoARay, ValidMask, INVALID_ID,
};

/// Builds a scene with a single user-geometry sphere, whose intersect callback
/// reports a hit a `t = 1.5` (the front of the unit sphere for the test ray)
/// and runs it through the filter chain. The registered intersect filter
/// captures `probe`/`calls` and either accepts (leaves valid = -1) or rejects
/// (writes valid = 0) based on `reject`.
fn trace_sphere_with_filter(reject: bool) -> (bool, Vec<u32>, usize) {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();

    let probe: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let probe_cb = probe.clone();
    let calls_cb = calls.clone();

    let mut sphere = common::user_sphere(&device);

    // Intersector: report a candidate hit at t=1.5, filter it, commit on
    // survivial.
    sphere.set_intersect_function::<_, (), IntersectContext>(
        move |args: &mut IntersectFunctionNArgs<'_, IntersectContext, ()>| {
            for i in 0..args.len() {
                if args.valid_n()[i] == 0 {
                    continue; // skip inactive rays
                }
                let mut ray = args.ray(i);
                let t = 1.5_f32;
                if t > ray.tnear && t < ray.tfar {
                    let mut hit = Hit {
                        Ng_x: 0.0,
                        Ng_y: 0.0,
                        Ng_z: -1.0,
                        u: 0.0,
                        v: 0.0,
                        primID: args.prim_id(),
                        geomID: args.geom_id(),
                        instID: [INVALID_ID], /* single-level, no instancing in this test
                                               * Initialize hit properties */
                    };
                    ray.tfar = t; // candidate distance the filter will see
                    if args.filter_intersection(&mut ray, &mut hit) {
                        args.commit_hit(i, &ray, &hit);
                    }
                }
            }
        },
    );

    sphere.set_intersect_filter_function::<_, (), IntersectContext>(
        move |_ray, hit, mut valid, _ctx, _user: Option<&()>| {
            calls_cb.fetch_add(1, Ordering::SeqCst);
            probe_cb.lock().unwrap().push(0xF11A ^ hit.prim_id(0));
            if reject {
                valid[0] = 0; // reject the hit
            }
        },
    );

    let sphere = sphere.commit();
    scene.attach_geometry(&sphere);
    scene.commit();

    // Make any dangling-stack UAF deterministic.
    common::clobber_stack();

    let ray = embree3::Ray::segment([0.0, 0.0, -2.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    let mut ctx = IntersectContext::coherent();
    let mut ray_hit = embree3::RayHit::from(ray);
    scene.intersect(&mut ctx, &mut ray_hit);

    // Bind the lock/clone to a local so the `MutexGuard` temporary is dropped at
    // this statement's end, not at the function block's end, where it would
    // outlive `probe`.
    let hit_valid = ray_hit.hit.is_valid();
    let recorded = probe.lock().unwrap().clone();
    let call_count = calls.load(Ordering::SeqCst);
    (hit_valid, recorded, call_count)
}

#[test]
fn filter_runs_for_user_geometry_and_accepts() {
    let (hit_valid, probe, calls) = trace_sphere_with_filter(false);
    assert!(
        calls >= 1,
        "the intersect filter must run via the user geometry"
    );
    assert_eq!(
        probe,
        vec![0xF11A ^ 0],
        "filter saw live captured state + primID 0"
    );
    assert!(hit_valid, "accepted hit must be committed");
}

#[test]
fn filter_runs_for_user_geometry_and_rejects() {
    let (hit_valid, _probe, calls) = trace_sphere_with_filter(true);
    assert!(
        calls >= 1,
        "the intersect filter must run via the user geometry"
    );
    assert!(!hit_valid, "rejected hit must NOT be committed");
}

/// Shadow-ray analogue: a user-geometry occluder whose occluded callback runs
/// the occlusion filter. When the filter rejects, the ray is NOT occluded.
fn trace_sphere_occlusion_with_filter(reject: bool) -> (bool, usize) {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cb = calls.clone();

    let mut sphere = common::user_sphere(&device);

    sphere.set_occluded_function::<_, (), IntersectContext>(
        move |args: &mut OccludedFunctionNArgs<'_, IntersectContext, ()>| {
            for i in 0..args.len() {
                if args.valid_n()[i] == 0 {
                    continue; // skip inactive rays
                }
                let mut ray = args.ray(i);
                let t = 1.5_f32;
                if t > ray.tnear && t < ray.tfar {
                    let mut hit = Hit {
                        Ng_x: 0.0,
                        Ng_y: 0.0,
                        Ng_z: -1.0,
                        u: 0.0,
                        v: 0.0,
                        primID: args.prim_id(),
                        geomID: args.geom_id(),
                        instID: [INVALID_ID], /* single-level, no instancing in this test
                                               * Initialize hit properties */
                    };
                    ray.tfar = t; // candidate distance the filter will see
                    if args.filter_occlusion(&mut ray, &mut hit) {
                        args.set_occluded(i);
                    }
                }
            }
        },
    );

    sphere.set_occluded_filter_function::<_, (), IntersectContext>(
        move |_ray, _hit, mut valid, _ctx, _user: Option<&()>| {
            calls_cb.fetch_add(1, Ordering::SeqCst);
            if reject {
                valid[0] = 0;
            }
        },
    );

    let sphere = sphere.commit();
    scene.attach_geometry(&sphere);
    scene.commit();

    common::clobber_stack();

    let ray = embree3::Ray::segment([0.0, 0.0, -2.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    let mut ctx = IntersectContext::coherent();
    let mut probe = ray;
    let occluded = scene.occluded(&mut ctx, &mut probe);

    (occluded, calls.load(Ordering::SeqCst))
}

#[test]
fn occlusion_filter_runs_and_occludes() {
    let (occluded, calls) = trace_sphere_occlusion_with_filter(false);
    assert!(
        calls >= 1,
        "the occlusion filter must run via the user geometry"
    );
    assert!(occluded, "surviving occluder must mark the ray occluded");
}

#[test]
fn occlusion_filter_runs_and_passes_through() {
    let (occluded, calls) = trace_sphere_occlusion_with_filter(true);
    assert!(
        calls >= 1,
        "the occlusion filter must run via the user geometry"
    );
    assert!(
        !occluded,
        "rejected occluder must NOT mark the ray occluded"
    );
}

#[test]
fn filter_rejecting_nearest_returns_farther_hit() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();

    let seen = Arc::new(AtomicUsize::new(0));
    let seen_cb = seen.clone();

    let mut sphere = common::user_sphere(&device);

    sphere.set_intersect_function::<_, (), IntersectContext>(
        move |args: &mut IntersectFunctionNArgs<'_, IntersectContext, ()>| {
            for i in 0..args.len() {
                if args.valid_n()[i] == 0 {
                    continue;
                }
                for &t in &[1.5_f32, 2.5_f32] {
                    let mut ray = args.ray(i);
                    if t > ray.tnear && t < ray.tfar {
                        let mut hit = Hit {
                            Ng_x: 0.0,
                            Ng_y: 0.0,
                            Ng_z: -1.0,
                            u: 0.0,
                            v: 0.0,
                            primID: args.prim_id(),
                            geomID: args.geom_id(),
                            instID: [INVALID_ID],
                        };
                        ray.tfar = t;
                        if args.filter_intersection(&mut ray, &mut hit) {
                            args.commit_hit(i, &ray, &hit);
                        }
                    }
                }
            }
        },
    );

    // Reject exactly the first hit the filter is shown; accept thereafter.
    sphere.set_intersect_filter_function::<_, (), IntersectContext>(
        move |_ray, _hit, mut valid, _ctx, _user: Option<&()>| {
            if seen_cb.fetch_add(1, Ordering::SeqCst) == 0 {
                valid[0] = 0; // reject the first (nearer) hit
            }
        },
    );

    let sphere = sphere.commit();
    scene.attach_geometry(&sphere);
    scene.commit();
    common::clobber_stack();

    let ray = embree3::Ray::segment([0.0, 0.0, -2.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    let mut ctx = IntersectContext::coherent();
    let mut ray_hit = embree3::RayHit::from(ray);
    scene.intersect(&mut ctx, &mut ray_hit);

    assert!(
        ray_hit.hit.is_valid(),
        "the farther hit should be committed"
    );
    // ray.tfar is the committed distance; the near hit (1.5) was rejected.
    assert!(
        (ray_hit.ray.tfar - 2.5).abs() < 1e-4,
        "committed hit must be the farther accepted one (t=2.5), got {}",
        ray_hit.ray.tfar
    );
}

#[test]
fn packet_filter_intersectio_n_rejects_per_lane() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cb = calls.clone();

    let mut sphere = common::user_sphere(&device);

    sphere.set_intersect_function::<_, (), IntersectContext>(
        move |args: &mut IntersectFunctionNArgs<'_, IntersectContext, ()>| {
            let n = args.len();
            let t = 1.5f32;
            let mut hits = vec![
                Hit {
                    primID: args.prim_id(),
                    geomID: args.geom_id(),
                    ..Default::default()
                };
                n
            ];
            let mut valid = vec![ValidMask::Valid as i32; n];
            let mut cached_tfar = vec![0.0; n];
            for i in 0..n {
                let ray = args.ray(i);
                cached_tfar[i] = ray.tfar;
                if args.valid_n()[i] != ValidMask::Invalid && t > ray.tnear && t < ray.tfar {
                    args.set_tfar(i, t); // candidate distance the filter will
                                         // see
                } else {
                    valid[i] = ValidMask::Invalid as i32;
                }
            }
            args.filter_intersection_n(&mut hits, &mut valid);
            for i in 0..n {
                if valid[i] != ValidMask::Invalid as i32 {
                    let r = args.ray(i); // tfar == t
                    args.commit_hit(i, &r, &hits[i]);
                } else {
                    args.set_tfar(i, cached_tfar[i]); // restore rejected /
                                                      // inative lanes
                }
            }
        },
    );

    sphere.set_intersect_filter_function::<_, (), IntersectContext>(
        move |ray, _hit, mut valid, _ctx, _user: Option<&()>| {
            calls_cb.fetch_add(1, Ordering::SeqCst);
            for i in 0..ray.len() {
                if ray.id(i) % 2 == 1 {
                    valid[i] = 0; // reject odd-id lanes
                }
            }
        },
    );

    let sphere = sphere.commit();
    scene.attach_geometry(&sphere);
    scene.commit();
    common::clobber_stack();

    // Four parallel rays down +z, all within the sphere's xy footprint; id = lane.
    let origins = [
        [-0.3, 0.0, -2.0],
        [-0.1, 0.0, -2.0],
        [0.1, 0.0, -2.0],
        [0.3, 0.0, -2.0],
    ];
    let mut ray4 = Ray4::new(origins, [[0.0, 0.0, 1.0]; 4]);
    for i in 0..4 {
        ray4.set_id(i, i as u32);
    }
    let mut rh = RayHit4::new(ray4);
    let valid = [-1i32; 4];
    let mut ctx = IntersectContext::coherent();
    scene.intersect4(&mut ctx, &mut rh, &valid);

    assert!(
        calls.load(Ordering::SeqCst) >= 1,
        "the filter must run for the packet"
    );
    for i in 0..4 {
        let hit = rh.hit.is_valid(i);
        if i % 2 == 0 {
            assert!(hit, "even-id lane {i} should hit (filter accepts)");
        } else {
            assert!(!hit, "odd-id lane {i} should be rejected by the filter");
        }
    }
}
