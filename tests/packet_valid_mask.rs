//! `ValidMaskN` gates which packet lanes embree processes: a lane masked
//! [`ValidMask::Invalid`] must be left untouched by `intersect{4,8,16}` /
//! `occluded{4,8,16}`. `point_query4` already has this coverage
//! (`tests/point_query_packet.rs`); this adds the ray-packet families, which is
//! the core contract of the typed mask.
mod common;

use embree3::{IntersectContext, Ray4, RayHit4, SoAHit as _, ValidMask, ValidMaskN};

/// Four identical rays straight down +z through (0.25, 0.25), each of which
/// hits `unit_triangle` on its own.
fn four_hitting_rays() -> Ray4 { Ray4::new([[0.25, 0.25, -1.0]; 4], [[0.0, 0.0, 1.0]; 4]) }

#[test]
fn intersect4_skips_disabled_lane() {
    let device = common::device();
    let tri = common::unit_triangle(&device).commit();
    let mut scene = device.create_scene().unwrap();
    scene.attach_geometry(&tri);
    scene.commit();

    let mut rh = RayHit4::new(four_hitting_rays());
    let mut valid = ValidMaskN::all_active();
    valid.lanes_mut()[3] = ValidMask::Invalid; // lane 3 disabled

    let mut ctx = IntersectContext::coherent();
    scene.intersect4(&mut ctx, &mut rh, &valid);

    for i in 0..3 {
        assert!(rh.hit.is_valid(i), "active lane {i} must hit the triangle");
        assert_eq!(
            rh.hit.geomID[i], 0,
            "active lane {i} hit the attached geometry"
        );
    }
    assert!(
        !rh.hit.is_valid(3),
        "disabled lane 3 must be left untouched (no hit written)"
    );
}

#[test]
fn occluded4_skips_disabled_lane() {
    let device = common::device();
    let tri = common::unit_triangle(&device).commit();
    let mut scene = device.create_scene().unwrap();
    scene.attach_geometry(&tri);
    scene.commit();

    let mut ray4 = four_hitting_rays();
    let mut valid = ValidMaskN::all_active();
    valid.lanes_mut()[3] = ValidMask::Invalid; // lane 3 disabled

    let mut ctx = IntersectContext::coherent();
    scene.occluded4(&mut ctx, &mut ray4, &valid);

    // embree marks an occluded lane by setting its `tfar` to -inf; an unprocessed
    // (disabled) lane keeps the initial `tfar` (INFINITY from `Ray4::new`).
    for i in 0..3 {
        assert_eq!(
            ray4.tfar[i],
            f32::NEG_INFINITY,
            "active lane {i} must be reported occluded"
        );
    }
    assert_ne!(
        ray4.tfar[3],
        f32::NEG_INFINITY,
        "disabled lane 3 must be left untouched"
    );
}
