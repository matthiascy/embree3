#![allow(dead_code)]

extern crate embree;
extern crate support;

use embree::{BufferUsage, Device, Format, IntersectContext, RayHitNp, RayNp, TriangleMeshBuilder};
use support::{rgba_to_u32, DebugState};

fn main() {
    let display = support::Display::new(512, 512, "triangle");

    let device = Device::new().unwrap();

    device.set_error_function(|error, message| {
        println!("Embree error {}: {}", error, message);
    });

    // Make a triangle
    let mut triangle = TriangleMeshBuilder::new(&device).unwrap();
    triangle
        .set_new_buffer::<[f32; 4]>(BufferUsage::VERTEX, 0, Format::FLOAT3, 16, 3)
        .unwrap()
        .copy_from_slice(&[
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]);
    triangle
        .set_new_buffer::<[u32; 3]>(BufferUsage::INDEX, 0, Format::UINT3, 12, 1)
        .unwrap()
        .copy_from_slice(&[[0, 1, 2]]);
    let triangle = triangle.commit();

    let mut scene = device.create_scene().unwrap();
    scene.attach_geometry(&triangle);
    scene.commit();

    let state = DebugState { scene, user: () };

    support::display::run(
        display,
        state,
        |_, _| {},
        move |image, _, _, state| {
            let mut intersection_ctx = IntersectContext::coherent();
            image.reinterpret_as_none_tiled();

            let img_dims = (image.width, image.height);
            // Render the scene
            for j in 0..img_dims.1 {
                let y = -(j as f32 + 0.5) / img_dims.1 as f32 + 0.5;

                // Try out streams of scanlines across x
                let mut rays = RayNp::new(img_dims.0 as usize);
                for (i, mut ray) in rays.iter_mut().enumerate() {
                    let x = (i as f32 + 0.5) / img_dims.0 as f32 - 0.5;
                    let dir_len = f32::sqrt(x * x + y * y + 1.0);
                    ray.set_org([0.0, 0.5, 2.0]);
                    ray.set_dir([x / dir_len, y / dir_len, -1.0 / dir_len]);
                }

                let mut ray_hit = RayHitNp::new(rays);
                state
                    .scene
                    .intersect_stream_soa(&mut intersection_ctx, &mut ray_hit);
                for (i, hit) in ray_hit
                    .hit
                    .iter()
                    .enumerate()
                    .filter(|(_i, h)| h.is_valid())
                {
                    let pixel = &mut image.pixels[i + (j * img_dims.0) as usize];
                    let uv = hit.uv();
                    *pixel = rgba_to_u32((uv[0] * 255.0) as u8, (uv[1] * 255.0) as u8, 0, 255);
                }
            }
        },
        |_| {},
    );
}
