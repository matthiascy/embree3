//! Proves dropping one Geometry clone does not free user-data the survivors
//! still use.
mod common;

use embree3::{Bounds, GeometryKind};
use std::sync::{Arc, Mutex};

struct UserData {
    magic: u32,
}

#[test]
fn dropping_one_clone_does_not_free_shared_user_data() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();

    let seen: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
    let seen_in_cb = seen.clone();

    let mut geom = device.create_geometry(GeometryKind::USER).unwrap();
    geom.set_user_primitive_count(1);
    // Owned data lives in the geometry's `Arc<GeometryShared>` (per-callback),
    // so it survives until the *last* clone drops.
    geom.set_bounds_function_owned::<_, UserData>(
        move |bounds: &mut Bounds, _prim, _time, user| {
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
        UserData { magic: 0x0BAD_F00D },
    );
    let geom = geom.commit();

    // Clone and drop one handle BEFORE the survivor is used (the committed
    // Geometry is Clone; the builder is not).
    let cloned = geom.clone();
    drop(cloned);

    scene.attach_geometry(&geom);
    common::clobber_stack();
    scene.commit(); // survivor's bounds callback reads the owned user data.

    assert_eq!(
        *seen.lock().unwrap(),
        Some(0x0BAD_F00D),
        "bounds callback must receive the owned user data, correctly typed, even if a clone was \
         dropped"
    );
}
