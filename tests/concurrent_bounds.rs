//! Proves a user-geometry bounds callback is invoked correctly under a
//! concurrent (multi-threaded) commit.
//!
//! With the default device the BVH build over many user primitives
//! parallelizes, so the bounds callback fires from several worker threads at
//! once. Soundness rests on the closure being `Fn` (recovered as a shared `&F`
//! in the trampoline, so concurrent invocations do not alias) and on its
//! captures being `Send + Sync`. This is a *positive* test: it passes both
//! before and after the fix under a normal run (UB need not manifest), but it
//! pins the API shape (a `!Send`/`FnMut` closure no longer compiles) and is
//! meant to be run under ThreadSanitizer to catch a regression to `&mut F`.

mod common;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use embree::{Bounds, GeometryKind};

#[test]
fn bounds_callback_runs_under_concurrent_commit() {
    let device = common::device(); // default threads => parallel build
    let mut scene = device.create_scene().unwrap();

    const N: u32 = 4096;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in_cb = calls.clone();

    let mut geom = device.create_geometry(GeometryKind::USER).unwrap();
    geom.set_user_primitive_count(N);
    geom.set_bounds_function::<_, ()>(move |bounds: &mut Bounds, prim, _time, _user| {
        // A distinct unit box per primitive so the builder has real work to
        // parallelize.
        let x = prim as f32;
        *bounds = Bounds {
            lower_x: x,
            lower_y: 0.0,
            lower_z: 0.0,
            align0: 0.0,
            upper_x: x + 1.0,
            upper_y: 1.0,
            upper_z: 1.0,
            align1: 0.0,
        };
        calls_in_cb.fetch_add(1, Ordering::Relaxed);
    });
    geom.commit();
    scene.attach_geometry(&geom);
    scene.commit();

    let n = calls.load(Ordering::Relaxed);
    assert!(
        n >= N as usize,
        "bounds callback must fire at least once per primitive (got {n}, want >= {N})"
    );
}
