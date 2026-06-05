//! Proves the device/scene callback trampolines recover their heap-owned
//! closure correctly -- regression guard for the userPtr-vs-`Mutex`
//! reinterpretation bug (the trampolines must cast the dedicated userPtr
//! straight to `*mut F`, since `set_*` passes `ErasedFn::as_ptr()`, not a
//! pointer to a control block).
//!
//! A device memory monitor is the easiest embree event to trigger
//! deterministically. The device is pinned to one build thread so the callback
//! fires single-threaded: the current trampoline forms `&mut F` from a shared
//! pointer, which would alias under concurrent firing so the test must not
//! provoke that. The captured state is an `Arc<AtomicUsize>` (Send + Sync) for
//! the same reason -- embree may invoke the monitor from a worker thread.

mod common;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use embree3::{Config, Device};

#[test]
fn memory_monitor_closure_is_invoked_with_live_state() {
    // One build thread => the memory monitor fires single-threaded.
    let device = Device::with_config(Config {
        threads: 1,
        ..Default::default()
    })
    .unwrap();

    let allocations = Arc::new(AtomicUsize::new(0));
    let seen = allocations.clone();
    device.set_memory_monitor_function(move |_bytes, _post| {
        seen.fetch_add(1, Ordering::Relaxed);
        true
    });

    // The closure now lives heap-owned inside the device; its registering stack
    // frame is gone. Clobber it to prove the trampoline does not depend on the
    // stack.
    common::clobber_stack();

    // Building and committing a scene allocates through the monitored allocator.
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    scene.attach_geometry(&tri);
    scene.commit();

    assert!(
        allocations.load(Ordering::Relaxed) > 0,
        "memory monitor must fire with a live, correctly-recovered closure (was the userPtr / \
         trampoline type mismatch reintroduced?)"
    );
    assert!(
        allocations.load(Ordering::Relaxed) > 1,
        "memory monitor should fire multiple times during scene build and commit, proving the \
         callback is still live after the first invocation"
    );
}
