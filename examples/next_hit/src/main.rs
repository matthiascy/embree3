//! Next-hit (multi-hit traversal) example, currently a placeholder stub.
//!
//! A full next-hit demo (iterating successive hits along a ray via an
//! intersection filter, mirroring embree's C++ `next_hit` tutorial) has not
//! been written against the current API yet. The previous content was a
//! copy-pasted `main` fragment with no imports or supporting code that never
//! compiled.
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
