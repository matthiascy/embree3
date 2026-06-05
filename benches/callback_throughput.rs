use std::collections::BTreeSet;

use criterion::{criterion_group, criterion_main, Criterion};
use embree::{BufferUsage, Device, Format, GeometryKind, IntersectContext, Ray, RayHit, Scene};

const GRID: usize = 512;

/// A scene whose cube carries a stateless intersect filter. Embree invokes that
/// filter once per hit candidate, from every traversal thread, so the filter's
/// *registration* is exactly the per-hit trampoline-dispatch cost this
/// benchmark isolates (a `Mutex` lock/unlock per call in the old design, a
/// plain lock-free `CallSite` read now).
///
/// The returned `Scene` is self-contained: it owns a `Device` clone and retains
/// a clone of the attached geometry, so no `Box::leak` / external owner is
/// needed to keep it `'static`.
fn build_filtered_scene() -> Scene<'static> {
    let device = Device::new().expect("failed to create embree device");
    let mut scene = device.create_scene().unwrap();

    // A unit cube in [0,1]^3 (12 triangles, real BVH). The ray grid in
    // `cast_grid` (origins at z=-1, x,y in [0,1)) marches +z and strikes the
    // cube's z=0 face, so *every* ray fires the filter, driving the per-hit
    // trampoline dispatch as hard as possible.
    let mut cube = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    cube.set_new_buffer::<[f32; 3]>(BufferUsage::VERTEX, 0, Format::FLOAT3, 12, 8)
        .unwrap()
        .copy_from_slice(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]);
    cube.set_new_buffer::<[u32; 3]>(BufferUsage::INDEX, 0, Format::UINT3, 12, 12)
        .unwrap()
        .copy_from_slice(&[
            [0, 1, 2],
            [1, 3, 2], // x = 0
            [4, 6, 5],
            [5, 6, 7], // x = 1
            [0, 4, 1],
            [1, 4, 5], // y = 0
            [2, 3, 6],
            [3, 7, 6], // y = 1
            [0, 2, 4],
            [2, 6, 4], // z = 0 (the face the rays hit)
            [1, 5, 3],
            [3, 5, 7], // z = 1
        ]);

    // Accept every hit (leave the valid mask untouched); the cost we measure is
    // the dispatch, not the body.
    cube.set_intersect_filter_function::<_, (), IntersectContext>(|_r, _h, _v, _c, _ud| {});

    let geom = cube.commit();
    scene.attach_geometry(&geom);
    scene.commit();
    scene
}

fn cast_grid(scene: &Scene, threads: usize) -> u64 {
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut hits = 0u64;
                    // Balanced row partition: every GRID row is covered exactly once even when GRID
                    // is not divisible by `threads`.
                    let start = t * GRID / threads;
                    let end = (t + 1) * GRID / threads;
                    for gy in start..end {
                        for gx in 0..GRID {
                            let mut ctx = IntersectContext::coherent();
                            let x = gx as f32 / GRID as f32;
                            let y = gy as f32 / GRID as f32;
                            let mut rh = RayHit::from(Ray::segment(
                                [x, y, -1.0],
                                [0.0, 0.0, 1.0],
                                0.0,
                                f32::INFINITY,
                            ));
                            scene.intersect(&mut ctx, &mut rh);
                            hits += rh.hit.is_valid() as u64;
                        }
                    }
                    hits
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn bench_filtered_traversal(c: &mut Criterion) {
    let scene = build_filtered_scene();
    let max = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let counts: BTreeSet<usize> = [1, 2, 4, 8, max]
        .into_iter()
        .filter(|&t| t <= max)
        .collect();
    let mut group = c.benchmark_group("filtered_traversal");
    for &threads in &counts {
        group.bench_function(format!("{threads}t"), |b| {
            b.iter(|| std::hint::black_box(cast_grid(&scene, threads)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_filtered_traversal);
criterion_main!(benches);
