//! Throughput of the `for_each_active_lane` / `IntersectLane` path against the
//! checked random-access loop, across packet widths and for occlusion.
//!
//! The lane handles call the `#[inline(always)] *_unchecked` gather/scatter
//! primitives, so on this path the per-lane bounds checks are removed
//! structurally (not merely left to the optimizer). Every `*_lane` case has a
//! matching `*_checked` case that does the same work through the public checked
//! API (`for i in 0..len { args.ray(i); args.commit_hit(i, ..) }`) as a
//! baseline. Both still include embree's traversal/dispatch cost, so use the
//! lane-vs-checked delta (not the absolute number) to gauge the bounds-check
//! overhead.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use embree3::{
    Bounds, Device, Geometry, GeometryKind, Hit, Hit16, Hit4, Hit8, IntersectContext,
    IntersectFunctionNArgs, OccludedFunctionNArgs, Ray16, Ray4, Ray8, RayHit16, RayHit4, RayHit8,
    Scene, ValidMaskN, INVALID_ID,
};

fn box_bounds(b: &mut Bounds, _p: u32, _t: u32, _u: Option<&()>) {
    b.lower_x = -0.5;
    b.lower_y = -0.5;
    b.lower_z = -0.5;
    b.upper_x = 0.5;
    b.upper_y = 0.5;
    b.upper_z = 0.5;
}

fn candidate_hit(prim: u32, geom: u32) -> Hit {
    Hit {
        Ng_x: 0.0,
        Ng_y: 0.0,
        Ng_z: -1.0,
        u: 0.0,
        v: 0.0,
        primID: prim,
        geomID: geom,
        instID: [INVALID_ID],
    }
}

fn attach(device: &Device, geom: Geometry<'static>) -> Scene<'static> {
    let mut s = device.create_scene().unwrap();
    s.attach_geometry(&geom); // scene retains a clone; `geom` may drop
    s.commit();
    s
}

/// Intersect callback driven through the unchecked lane handles.
fn lane_scene(device: &Device) -> Scene<'static> {
    let mut g = device.create_geometry(GeometryKind::USER).unwrap();
    g.set_primitive_count(1);
    g.set_bounds_function::<_, ()>(box_bounds);
    g.set_intersect_function::<_, ()>(|args: &mut IntersectFunctionNArgs<'_, ()>| {
        args.for_each_active_lane(|mut lane| {
            let mut ray = lane.ray();
            let t = 1.5_f32;
            if t > ray.tnear && t < ray.tfar {
                let mut hit = candidate_hit(lane.prim_id(), lane.geom_id());
                ray.tfar = t;
                if lane.filter_intersection(&mut ray, &mut hit) {
                    lane.commit_hit(&ray, &hit);
                }
            }
        });
    });
    attach(device, g.commit())
}

/// Same work via the public checked random-access API (baseline).
fn checked_scene(device: &Device) -> Scene<'static> {
    let mut g = device.create_geometry(GeometryKind::USER).unwrap();
    g.set_primitive_count(1);
    g.set_bounds_function::<_, ()>(box_bounds);
    g.set_intersect_function::<_, ()>(|args: &mut IntersectFunctionNArgs<'_, ()>| {
        for i in 0..args.len() {
            if args.valid_n()[i] == 0 {
                continue;
            }
            let mut ray = args.ray(i);
            let t = 1.5_f32;
            if t > ray.tnear && t < ray.tfar {
                let mut hit = candidate_hit(args.prim_id(), args.geom_id());
                ray.tfar = t;
                if args.filter_intersection(&mut ray, &mut hit) {
                    args.commit_hit(i, &ray, &hit);
                }
            }
        }
    });
    attach(device, g.commit())
}

/// Occluded callback driven through the unchecked lane handles.
fn occluded_lane_scene(device: &Device) -> Scene<'static> {
    let mut g = device.create_geometry(GeometryKind::USER).unwrap();
    g.set_primitive_count(1);
    g.set_bounds_function::<_, ()>(box_bounds);
    g.set_occluded_function::<_, ()>(|args: &mut OccludedFunctionNArgs<'_, ()>| {
        args.for_each_active_lane(|mut lane| {
            let mut ray = lane.ray();
            let t = 1.5_f32;
            if t > ray.tnear && t < ray.tfar {
                let mut hit = candidate_hit(lane.prim_id(), lane.geom_id());
                ray.tfar = t;
                if lane.filter_occlusion(&mut ray, &mut hit) {
                    lane.set_occluded();
                }
            }
        });
    });
    attach(device, g.commit())
}

/// Occlusion via the public checked random-access API (baseline).
fn occluded_checked_scene(device: &Device) -> Scene<'static> {
    let mut g = device.create_geometry(GeometryKind::USER).unwrap();
    g.set_primitive_count(1);
    g.set_bounds_function::<_, ()>(box_bounds);
    g.set_occluded_function::<_, ()>(|args: &mut OccludedFunctionNArgs<'_, ()>| {
        for i in 0..args.len() {
            if args.valid_n()[i] == 0 {
                continue;
            }
            let mut ray = args.ray(i);
            let t = 1.5_f32;
            if t > ray.tnear && t < ray.tfar {
                let mut hit = candidate_hit(args.prim_id(), args.geom_id());
                ray.tfar = t;
                if args.filter_occlusion(&mut ray, &mut hit) {
                    args.set_occluded(i);
                }
            }
        }
    });
    attach(device, g.commit())
}

fn bench(c: &mut Criterion) {
    let device = Device::new().expect("device");
    let lane = lane_scene(&device);
    let checked = checked_scene(&device);
    let occl = occluded_lane_scene(&device);
    let occl_checked = occluded_checked_scene(&device);

    let mut g = c.benchmark_group("lane_iter");

    g.throughput(Throughput::Elements(4));
    g.bench_function("intersect4_lane", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<4>::all_active();
        b.iter(|| {
            let mut rh = RayHit4 {
                ray: Ray4::new([[0.0, 0.0, -2.0]; 4], [[0.0, 0.0, 1.0]; 4]),
                hit: Hit4::new(),
            };
            lane.intersect4(&mut ctx, &mut rh, &mask);
            black_box(&rh);
        });
    });
    g.bench_function("intersect4_checked", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<4>::all_active();
        b.iter(|| {
            let mut rh = RayHit4 {
                ray: Ray4::new([[0.0, 0.0, -2.0]; 4], [[0.0, 0.0, 1.0]; 4]),
                hit: Hit4::new(),
            };
            checked.intersect4(&mut ctx, &mut rh, &mask);
            black_box(&rh);
        });
    });

    g.throughput(Throughput::Elements(8));
    g.bench_function("intersect8_lane", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<8>::all_active();
        b.iter(|| {
            let mut rh = RayHit8 {
                ray: Ray8::new([[0.0, 0.0, -2.0]; 8], [[0.0, 0.0, 1.0]; 8]),
                hit: Hit8::new(),
            };
            lane.intersect8(&mut ctx, &mut rh, &mask);
            black_box(&rh);
        });
    });

    g.bench_function("intersect8_checked", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<8>::all_active();
        b.iter(|| {
            let mut rh = RayHit8 {
                ray: Ray8::new([[0.0, 0.0, -2.0]; 8], [[0.0, 0.0, 1.0]; 8]),
                hit: Hit8::new(),
            };
            checked.intersect8(&mut ctx, &mut rh, &mask);
            black_box(&rh);
        });
    });

    g.throughput(Throughput::Elements(16));
    g.bench_function("intersect16_lane", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<16>::all_active();
        b.iter(|| {
            let mut rh = RayHit16 {
                ray: Ray16::new([[0.0, 0.0, -2.0]; 16], [[0.0, 0.0, 1.0]; 16]),
                hit: Hit16::new(),
            };
            lane.intersect16(&mut ctx, &mut rh, &mask);
            black_box(&rh);
        });
    });
    g.bench_function("intersect16_checked", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<16>::all_active();
        b.iter(|| {
            let mut rh = RayHit16 {
                ray: Ray16::new([[0.0, 0.0, -2.0]; 16], [[0.0, 0.0, 1.0]; 16]),
                hit: Hit16::new(),
            };
            checked.intersect16(&mut ctx, &mut rh, &mask);
            black_box(&rh);
        });
    });

    g.throughput(Throughput::Elements(8));
    g.bench_function("occluded8_lane", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<8>::all_active();
        b.iter(|| {
            let mut ray = Ray8::new([[0.0, 0.0, -2.0]; 8], [[0.0, 0.0, 1.0]; 8]);
            occl.occluded8(&mut ctx, &mut ray, &mask);
            black_box(&ray);
        });
    });
    g.bench_function("occluded8_checked", |b| {
        let mut ctx = IntersectContext::coherent();
        let mask = ValidMaskN::<8>::all_active();
        b.iter(|| {
            let mut ray = Ray8::new([[0.0, 0.0, -2.0]; 8], [[0.0, 0.0, 1.0]; 8]);
            occl_checked.occluded8(&mut ctx, &mut ray, &mask);
            black_box(&ray);
        });
    });

    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
