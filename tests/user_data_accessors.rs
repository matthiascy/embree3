//! `callback_data` / `callback_data_mut` round-trip for the per-callback owned
//! data binding. These read through the lock-free `CallbackOwners` (an `Any`
//! downcast, not the raw embree pointer), so this also guards against the
//! type-confusion regression the old single-value API was prone to.

mod common;

use embree3::{Bounds, CbKind, GeometryKind};

#[derive(Debug, PartialEq)]
struct UserData {
    magic: u32,
}

#[test]
fn callback_data_roundtrips_and_is_type_checked() {
    let device = common::device();
    let mut geom = device.create_geometry(GeometryKind::USER).unwrap();

    // Nothing bound yet.
    assert!(geom.callback_data::<UserData>(CbKind::UserBounds).is_none());

    // Bind owned data to the bounds callback. The closure body is irrelevant
    // here (the scene is never committed); we only exercise the getters.
    geom.set_bounds_function_owned::<_, UserData>(
        |_b: &mut Bounds, _p, _t, _u: Option<&UserData>| {},
        UserData { magic: 0xCAFE },
    );

    // Shared read returns the value, correctly typed.
    assert_eq!(
        geom.callback_data::<UserData>(CbKind::UserBounds)
            .map(|u| u.magic),
        Some(0xCAFE)
    );

    // A mismatched type yields None (Any downcast fails), never a misread.
    assert!(geom.callback_data::<u64>(CbKind::UserBounds).is_none());

    // A different slot has nothing bound.
    assert!(geom
        .callback_data::<UserData>(CbKind::IntersectFilter)
        .is_none());

    // Mutable access goes through `&mut self` (the unique builder).
    if let Some(u) = geom.callback_data_mut::<UserData>(CbKind::UserBounds) {
        u.magic = 0xBEEF;
    }
    assert_eq!(
        geom.callback_data::<UserData>(CbKind::UserBounds)
            .map(|u| u.magic),
        Some(0xBEEF)
    );
}
