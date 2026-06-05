//! User-geometry example, currently a placeholder stub.
//!
//! A full user-geometry demo (analytic spheres with bounds / intersect /
//! occluded callbacks and a render loop, mirroring embree's C++ `user_geometry`
//! tutorial) has not been ported to the post-soundness-fix API yet. The earlier
//! WIP here relied on the removed `Option<&mut D>` callback shape and an
//! unsound `scene: &'a mut Scene<'a>` lifetime tangle, so it was removed rather
//! than force-compiled.
//!
//! This stub keeps the workspace building. It sets up a device and scene but
//! renders nothing.

use embree3::Device;

fn main() {
    let device = Device::new().unwrap();
    device.set_error_function(|err, msg| {
        eprintln!("{}: {}", err, msg);
    });
    let _scene = device.create_scene().unwrap();
}
