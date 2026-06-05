//! Proves the geometry user-data pointer reaches a callback with the correct
//! type.
mod common;

use std::sync::{Arc, Mutex};

use embree3::{Bounds, GeometryKind};

struct UserData {
    magic: u32,
}

#[test]
fn user_data_reaches_bounds_callback_correctly_typed() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();

    // Channel the callback writes its observation of `user.magic` into.
    let seen: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
    let seen_in_cb = seen.clone();

    let mut geom = device.create_geometry(GeometryKind::USER).unwrap();
    geom.set_user_primitive_count(1);
    // Per-callback owned data, bound at the setter (replaces the old single-value
    // `set_owned_user_data` + separate `set_bounds_function`).
    geom.set_bounds_function_owned::<_, UserData>(
        move |bounds: &mut Bounds, _prim, _time, user| {
            // A single unit box so the BVH builder is happy.
            *bounds = Bounds {
                lower_x: 0.0,
                lower_y: 0.0,
                lower_z: 0.0,
                align0: 0.0,
                upper_x: 1.0,
                upper_y: 1.0,
                upper_z: 1.0,
                align1: 0.0,
            };
            *seen_in_cb.lock().unwrap() = user.map(|u| u.magic);
        },
        UserData { magic: 0x1234_5678 },
    );

    let geom = geom.commit();
    scene.attach_geometry(&geom);

    common::clobber_stack();
    scene.commit(); // triggers the bounds callback, which should write into
                    // `seen`.

    assert_eq!(
        *seen.lock().unwrap(),
        Some(0x1234_5678),
        "bounds callback must receive the owned user data, correctly typed"
    );
}
