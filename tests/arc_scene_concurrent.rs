//! Concurrent read-only queries on a committed scene shared via `Arc<Scene>`.
//!
//! Making `Scene` `Send + Sync` and shareable through `Arc<Scene>` (instead of
//! the old `rtcRetainScene`-based `Clone`, which aliased one native scene and
//! broke `&mut` exclusivity) exists for exactly one reason: many threads can
//! `intersect`/`occluded` the *same* committed scene at once, because queries
//! take `&self`. This test exercises that path; run under ThreadSanitizer to
//! prove the shared `&self` query is actually race-free, not merely race-free
//! by luck.
mod common;

use std::{sync::Arc, thread};

use embree3::Scene;

#[test]
fn arc_scene_concurrent_intersect_is_race_free() {
    let device = common::device();
    let tri = common::unit_triangle(&device).commit();
    let mut scene = device.create_scene().unwrap();
    scene.attach_geometry(&tri);
    scene.commit();

    // Share the *committed* scene; `Arc` hands out only `&Scene`, so no thread
    // can mutate or re-commit it (that would need `&mut Scene`) -- see the
    // `scene_no_mutation_through_arc` compile-fail test.
    let scene: Arc<Scene> = Arc::new(scene);

    const THREADS: usize = 8;
    const RAYS_PER_THREAD: u32 = 2000;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let s = Arc::clone(&scene);
            thread::spawn(move || {
                let mut hits = 0u32;
                for _ in 0..RAYS_PER_THREAD {
                    // `&Arc<Scene>` derefs to the `&Scene` that `intersect` (via
                    // `cast_center_ray`) takes -- a shared, read-only query.
                    let rh = common::cast_center_ray(&s);
                    if rh.hit.is_valid() {
                        hits += 1;
                    }
                }
                hits
            })
        })
        .collect();

    for h in handles {
        assert_eq!(
            h.join().unwrap(),
            RAYS_PER_THREAD,
            "every center ray must hit the triangle, from every thread"
        );
    }
}
