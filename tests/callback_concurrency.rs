mod common;
use embree::{IntersectContext, Ray, RayHit};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[test]
fn concurrent_filter_invocation_is_race_free() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let mut tri = common::unit_triangle(&device);
    let hits = Arc::new(AtomicUsize::new(0));
    let captured = hits.clone();
    tri.set_intersect_filter_function(
        move |_r, _h, _v, _c: &mut IntersectContext, _ud: Option<&()>| {
            captured.fetch_add(1, Ordering::Relaxed);
        },
    );
    let geom = tri.commit();
    scene.attach_geometry(&geom);
    scene.commit();
    common::clobber_stack();

    const N: usize = 200_000;
    let scene_ref = &scene;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(move || {
                for _ in 0..N {
                    let mut ctx = IntersectContext::coherent();
                    let mut rh = RayHit::from(Ray::segment(
                        [0.25, 0.25, -1.0],
                        [0.0, 0.0, 1.0],
                        0.0,
                        f32::INFINITY,
                    ));
                    scene_ref.intersect(&mut ctx, &mut rh);
                }
            });
        }
    });
    assert!(
        hits.load(Ordering::Relaxed) >= N * threads,
        "every ray invokes the filter at least once"
    );
}
