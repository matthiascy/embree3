//! Soundness proofs for the GeometryBuilder/Geometry typestate split.
//!
//! The invariant: we may obtain a mutable `GeometryBuilder` (via `try_edit`)
//! only when we are the **sole owner** of the geometry. A geometry attached to
//! a scene is shared (the scene retains a clone), so it cannot be edited until
//! detached; this is exactly embree's "do not modify an object that is in
//! use", enforced by the `Arc` refcount. The compile-fail doctests on
//! `GeometryBuilder` prove it is `!Clone`/`!Sync`.
mod common;

#[test]
fn sole_owner_can_edit_and_round_trip() {
    let device = common::device();
    let geom = common::unit_triangle(&device).commit(); // sole owner (Arc strong_count == 1)
    let builder = geom
        .try_edit()
        .expect("a sole-owner geometry must be editable");
    let _committed_again = builder.commit(); // builder -> committed Geometry
                                             // round-trips
}

#[test]
fn attached_geometry_cannot_be_edited() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let geom = common::unit_triangle(&device).commit();
    scene.attach_geometry(&geom); // scene retains a clone -> strong_count >= 2

    assert!(
        geom.try_edit().is_err(),
        "a geometry attached to a scene must not be editable (it is shared)"
    );
}

#[test]
fn detach_restores_editability() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let geom = common::unit_triangle(&device).commit();
    let id = scene.attach_geometry(&geom);

    // Shared while attached, even a fresh clone cannot edit it.
    assert!(
        geom.clone().try_edit().is_err(),
        "attached geometry is not editable"
    );

    scene.detach_geometry(id); // scene drops its clone -> strong_count back to 1
    assert!(
        geom.try_edit().is_ok(),
        "after detaching from every scene, the sole owner can edit again"
    );
}

#[test]
fn other_clone_blocks_editing() {
    let device = common::device();
    let geom = common::unit_triangle(&device).commit();
    let _alias = geom.clone(); // a second wrapper clone -> strong_count == 2

    assert!(
        geom.try_edit().is_err(),
        "a geometry with another live clone is shared and must not be editable"
    );
}

/// The dynamic scene-mediated API (added for the interactive examples):
/// visibility and per-geometry commit through `&mut Scene`, which excludes
/// concurrent traversal.
#[test]
fn scene_disable_then_enable_geometry() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    let id = scene.attach_geometry(&tri);
    scene.commit();
    assert!(
        common::cast_center_ray(&scene).hit.is_valid(),
        "the triangle should be hit while enabled"
    );

    scene.disable_geometry(id);
    scene.commit();
    assert!(
        !common::cast_center_ray(&scene).hit.is_valid(),
        "a disabled geometry must not be hit"
    );

    scene.enable_geometry(id);
    scene.commit();
    assert!(
        common::cast_center_ray(&scene).hit.is_valid(),
        "a re-enabled geometry should be hit again"
    );
}
