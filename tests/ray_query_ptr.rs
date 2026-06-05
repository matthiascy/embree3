//! Pointer-stream queries (`rtcIntersect1Mp` / `rtcOccluded1Mp`): an array of M
//! pointers to *scattered* single rays (as opposed to the contiguous `1M`/`NM`
//! stream). Proves results are written back per-ray and that hit/miss is
//! correct.
mod common;

use embree3::{IntersectContext, Ray, RayHit};

fn triangle_scene() -> (embree3::Device, embree3::Scene<'static>) {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    scene.attach_geometry(&tri);
    scene.commit();
    (device, scene)
}

#[test]
fn intersect1mp_resolves_each_scattered_rayhit() {
    let (_device, scene) = triangle_scene();

    // Scattered (separately allocated) ray/hits: one through the triangle at
    // (0.25, 0.25), one that misses far away.
    let mut hit = RayHit::from(Ray::segment(
        [0.25, 0.25, -1.0],
        [0.0, 0.0, 1.0],
        0.0,
        f32::INFINITY,
    ));
    let mut miss = RayHit::from(Ray::segment(
        [5.0, 5.0, -1.0],
        [0.0, 0.0, 1.0],
        0.0,
        f32::INFINITY,
    ));

    {
        let mut rays: Vec<&mut RayHit> = vec![&mut hit, &mut miss];
        let mut ctx = IntersectContext::coherent();
        scene.intersect1mp(&mut ctx, &mut rays);
    } // drop the &mut borrows so we can read the results back

    assert!(hit.hit.is_valid(), "scattered ray 0 must hit the triangle");
    assert!(!miss.hit.is_valid(), "scattered ray 1 must miss");
}

#[test]
fn occluded1mp_marks_each_scattered_ray() {
    let (_device, scene) = triangle_scene();

    let mut blocked = Ray::segment([0.25, 0.25, -1.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    let mut clear = Ray::segment([5.0, 5.0, -1.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);

    {
        let mut rays: Vec<&mut Ray> = vec![&mut blocked, &mut clear];
        let mut ctx = IntersectContext::coherent();
        scene.occluded1mp(&mut ctx, &mut rays);
    }

    // embree signals occlusion by setting tfar = -inf.
    assert_eq!(
        blocked.tfar,
        f32::NEG_INFINITY,
        "occluded ray must be marked (tfar = -inf)"
    );
    assert_ne!(
        clear.tfar,
        f32::NEG_INFINITY,
        "non-occluded ray must be left unchanged"
    );
}
