//! `get_user_data` / `get_user_data_mut` round-trip after the user-data pointer
//! unification. These read through the locked `GeometryData` (not the raw embree
//! pointer), so this also guards against the type-confusion regression (SB-2).

mod common;

use embree::GeometryKind;

#[derive(Debug, PartialEq)]
struct UserData {
    magic: u32,
}

#[test]
fn get_user_data_roundtrips_and_is_type_checked() {
    let device = common::device();
    let mut geom = device.create_geometry(GeometryKind::USER).unwrap();

    // Nothing set yet.
    assert!(geom.get_user_data::<UserData>().is_none());

    geom.set_owned_user_data(UserData { magic: 0xCAFE });

    // Shared read returns the value, correctly typed.
    assert_eq!(geom.get_user_data::<UserData>().map(|u| u.magic), Some(0xCAFE));

    // A mismatched type yields None (type_id check), never a misread.
    assert!(geom.get_user_data::<u64>().is_none());

    // Mutable access goes through `&mut self`.
    if let Some(u) = geom.get_user_data_mut::<UserData>() {
        u.magic = 0xBEEF;
    }
    assert_eq!(geom.get_user_data::<UserData>().map(|u| u.magic), Some(0xBEEF));
}
