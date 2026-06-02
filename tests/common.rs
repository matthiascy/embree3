//! Shared helpers for the FFI soundness proof tests.
#![allow(dead_code)]

use embree::{
    BufferUsage, Device, Format, Geometry, GeometryKind, IntersectContext, Ray, RayHit, Scene,
};

pub fn device() -> Device {
    Device::new().expect("failed to create embree device (is EMBREE_DIR/LD_LIBRARY_PATH set?)")
}

pub fn unit_triangle(device: &Device) -> Geometry<'static> {
    let mut tri = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    tri.set_new_buffer(BufferUsage::VERTEX, 0, Format::FLOAT3, 3 * 4, 3)
        .unwrap()
        .view_mut::<[f32; 3]>()
        .unwrap()
        .copy_from_slice(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    tri.set_new_buffer(BufferUsage::INDEX, 0, Format::UINT3, 3 * 4, 1)
        .unwrap()
        .view_mut::<[u32; 3]>()
        .unwrap()
        .copy_from_slice(&[[0, 1, 2]]);
    tri
}

/// Provokes embree into reporting an error through the device's error function.
/// Binding a vertex buffer to an out-of-range slot makes embree report
/// `RTC_ERROR_INVALID_ARGUMENT` via the error callback (verified empirically;
/// it does not abort, unlike e.g. committing a user geometry with no bounds
/// function). Used by the default-error-reporter tests.
pub fn trigger_embree_error(device: &Device) {
    let mut geom = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    let _ = geom.set_new_buffer(BufferUsage::VERTEX, 100, Format::FLOAT3, 3 * 4, 3);
}

/// Casts one ray straight down +z at (0.25, 0.25), which hits `unit_triangle`.
/// Returns the committed `RayHit` so callers can inspect `hit.geomID` etc.
pub fn cast_center_ray(scene: &Scene) -> RayHit {
    let ray = Ray::segment([0.25, 0.25, -1.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    let mut ctx = IntersectContext::coherent();
    let mut ray_hit = RayHit::from(ray);
    scene.intersect(&mut ctx, &mut ray_hit);
    ray_hit
}

/// Overwrites ~64 KiB of fresh stack so that any pointer left dangling into a
/// previously-returned stack frame now reads clobbered bytes. Call this between
/// registering a callback and tracing, to make the use-after-free deterministic
/// instead of "works by luck because the bytes are still there".
#[inline(never)]
pub fn clobber_stack() {
    let mut buf = [0xA5u8; 64 * 1024];
    // Touch every page so the writes are not optimised away.
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i as u8) ^ 0x5A;
    }
    std::hint::black_box(&buf);
}
