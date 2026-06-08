#![allow(dead_code)]

extern crate embree3;
extern crate support;

use embree3::{
    BufferUsage, Device, Format, Geometry, IntersectContext, QuadMeshBuilder, Ray, RayHit,
    TriangleMeshBuilder, INVALID_ID,
};
use glam::Vec3;
use support::*;

const DISPLAY_WIDTH: u32 = 512;
const DISPLAY_HEIGHT: u32 = 512;

/// Per-vertex colors, stored as `Vec3fa` (stride 16) so the last element is
/// readable with a 16-byte SSE load: exactly how embree's C++ tutorial lays
/// them out. Being `'static`, they can be shared zero-copy into a `'static`
/// geometry (the windowed event loop requires `'static` state).
static VERTEX_COLORS: [[f32; 4]; 8] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
];

fn bytes_of<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn make_cube(device: &Device) -> Geometry<'static> {
    let mut mesh = TriangleMeshBuilder::new(device).unwrap();
    {
        mesh.set_new_buffer::<[f32; 3]>(BufferUsage::VERTEX, 0, Format::FLOAT3, 12, 8)
            .unwrap()
            .copy_from_slice(&[
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]);
        mesh.set_new_buffer::<[u32; 3]>(BufferUsage::INDEX, 0, Format::UINT3, 12, 12)
            .unwrap()
            .copy_from_slice(&[
                // left side
                [0, 1, 2],
                [1, 3, 2],
                // right side
                [4, 6, 5],
                [5, 6, 7],
                // bottom side
                [0, 4, 1],
                [1, 4, 5],
                // top side
                [2, 3, 6],
                [3, 7, 6],
                // front side
                [0, 2, 4],
                [2, 6, 4],
                // back side
                [1, 5, 3],
                [3, 5, 7],
            ]);

        mesh.set_vertex_attribute_count(1);
        // Zero-copy: bind the `'static` color array directly
        // (`rtcSetSharedGeometryBuffer`). `FLOAT3` reads the first 12 bytes of each
        // 16-byte `Vec3fa`; the trailing float makes the last element 16-byte
        // SSE-readable, so the bound slice is exactly `8 * 16 = 128` bytes.
        mesh.set_shared_buffer(
            BufferUsage::VERTEX_ATTRIBUTE,
            0,
            Format::FLOAT3,
            bytes_of(&VERTEX_COLORS[..]),
            16,
            8,
        )
        .unwrap();
    }
    mesh.commit()
}

fn make_ground_plane(device: &Device) -> Geometry<'static> {
    let mut mesh = QuadMeshBuilder::new(device).unwrap();
    {
        mesh.set_new_buffer::<[f32; 4]>(BufferUsage::VERTEX, 0, Format::FLOAT3, 16, 4)
            .unwrap()
            .copy_from_slice(&[
                [-10.0, -2.0, -10.0, 0.0],
                [-10.0, -2.0, 10.0, 0.0],
                [10.0, -2.0, 10.0, 0.0],
                [10.0, -2.0, -10.0, 0.0],
            ]);
        mesh.set_new_buffer::<[u32; 4]>(BufferUsage::INDEX, 0, Format::UINT4, 16, 1)
            .unwrap()
            .copy_from_slice(&[[0, 1, 2, 3]]);
    }
    mesh.commit()
}

type State = DebugState<UserState>;

struct UserState {
    ground_id: u32,
    cube_id: u32,
    face_colors: Vec<[f32; 3]>,
    light_dir: Vec3,
}

fn main() {
    let display = Display::new(DISPLAY_WIDTH, DISPLAY_HEIGHT, "triangle geometry");
    let device = Device::new().unwrap();
    device.set_error_function(|err, msg| {
        println!("{}: {}", err, msg);
    });
    let scene = device.create_scene().unwrap();
    let user_state = UserState {
        face_colors: vec![
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        ground_id: INVALID_ID,
        cube_id: INVALID_ID,
        light_dir: Vec3::new(1.0, 1.0, 1.0).normalize(),
    };

    let mut state = State {
        scene,
        user: user_state,
    };

    let cube = make_cube(&device);
    let ground = make_ground_plane(&device);
    state.user.cube_id = state.scene.attach_geometry(&cube);
    state.user.ground_id = state.scene.attach_geometry(&ground);

    state.scene.commit();

    display::run(display, state, move |_, _| {}, render_frame, |_| {});
}

// Task that renders a single pixel.
fn render_pixel(x: u32, y: u32, _time: f32, camera: &Camera, state: &State) -> u32 {
    let mut ctx = IntersectContext::coherent();
    let dir = camera.ray_dir((x as f32 + 0.5, y as f32 + 0.5));
    let mut ray_hit = RayHit::from_ray(Ray::segment(
        camera.pos.into(),
        dir.into(),
        0.0,
        f32::INFINITY,
    ));
    state.scene.intersect(&mut ctx, &mut ray_hit);
    let mut pixel = 0;
    if ray_hit.hit.is_valid() {
        let diffuse = if ray_hit.hit.geomID == state.user.ground_id {
            glam::vec3(0.6, 0.6, 0.6)
        } else {
            glam::Vec3::from(state.user.face_colors[ray_hit.hit.primID as usize])
        };

        let mut shadow_ray = Ray::segment(
            ray_hit.ray.hit_point(),
            state.user.light_dir.into(),
            0.001,
            f32::INFINITY,
        );

        // Check if the shadow ray is occluded.
        let color = if !state.scene.occluded(&mut ctx, &mut shadow_ray) {
            diffuse
        } else {
            diffuse * 0.5
        };

        pixel = rgba_to_u32(
            (color.x * 255.0) as u8,
            (color.y * 255.0) as u8,
            (color.z * 255.0) as u8,
            255,
        );
    }
    pixel
}

fn render_frame(frame: &mut TiledImage, camera: &Camera, time: f32, state: &mut State) {
    frame.par_tiles_mut().for_each(|tile| {
        tile.pixels.iter_mut().enumerate().for_each(|(i, pixel)| {
            let x = tile.x + (i % tile.w as usize) as u32;
            let y = tile.y + (i / tile.w as usize) as u32;
            *pixel = render_pixel(x, y, time, camera, state);
        });
    });
}
