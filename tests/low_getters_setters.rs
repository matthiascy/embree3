//! Small getter/setter wrappers: `Scene::get_linear_bounds`
//! (`rtcGetSceneLinearBounds`) and `GeometryBuilder::set_max_radius_scale`
//! (`rtcSetGeometryMaxRadiusScale`).
mod common;

use embree3::{Error, GeometryKind};

#[test]
fn get_linear_bounds_equals_static_bounds_for_non_motion_scene() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    scene.attach_geometry(&tri);
    scene.commit();

    let lb = scene.get_linear_bounds();
    let b = scene.get_bounds();

    // No motion blur: the t=0 and t=1 AABBs both equal the static bounds.
    for (lin, name) in [(lb.bounds0, "bounds0"), (lb.bounds1, "bounds1")] {
        assert_eq!(lin.lower_x, b.lower_x, "{name}.lower_x");
        assert_eq!(lin.lower_y, b.lower_y, "{name}.lower_y");
        assert_eq!(lin.lower_z, b.lower_z, "{name}.lower_z");
        assert_eq!(lin.upper_x, b.upper_x, "{name}.upper_x");
        assert_eq!(lin.upper_y, b.upper_y, "{name}.upper_y");
        assert_eq!(lin.upper_z, b.upper_z, "{name}.upper_z");
    }
}

#[test]
fn set_max_radius_scale_is_rejected_without_min_width_feature() {
    // The min-width feature is off in default embree builds, so embree rejects
    // setting a max radius scale with INVALID_OPERATION. With
    // `EMBREE_MIN_WIDTH` enabled it would succeed for a scale >= 1.0.
    let device = common::device();
    let mut geom = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    assert_eq!(
        geom.set_max_radius_scale(4.0),
        Err(Error::INVALID_OPERATION),
        "min-width is disabled by default, so this must be rejected"
    );
}
