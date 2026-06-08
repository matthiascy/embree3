//! A committed `Scene` shared across threads via `Arc<Scene>` (the documented
//! replacement for the old aliasing `Clone`) must be impossible to commit or
//! mutate: `commit`/`set_flags`/`set_build_quality`/progress/per-frame edits
//! all take `&mut self`, and `Arc` hands out only `&`. This is what makes the
//! "share for concurrent read-only queries" model sound -- no thread can mutate
//! the scene while others query it. Committing through the `Arc` must not
//! compile.
use std::sync::Arc;

use embree3::Device;

fn main() {
    let device = Device::new().unwrap();
    let scene = Arc::new(device.create_scene().unwrap());
    // ERROR: cannot borrow data in an `Arc` as mutable (`commit` needs `&mut
    // self`).
    scene.commit();
}
