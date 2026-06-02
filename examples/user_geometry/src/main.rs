use embree::{
    AlignedVector, Device, Geometry, GeometryKind, IntersectContext, RayHitN, Scene, SoARay,
    ValidMask, ValidityN,
};
use glam::{Mat3A, Vec3, Vec3A};
use support::{Display, Mode, DEFAULT_DISPLAY_HEIGHT, DEFAULT_DISPLAY_WIDTH};

// #[derive(Debug, Copy, Clone)]
// pub enum QuadraticSolution {
//     None,
//     One(f32),
//     Two(f32, f32),
// }
//
// pub fn solve_quadratic(a: f32, b: f32, c: f32) -> QuadraticSolution {
//     let discriminant = b * b - 4.0 * a * c;
//     let rcp_2a = 0.5 * rcp(a);
//     if discriminant < 0.0 {
//         QuadraticSolution::None
//     } else if discriminant == 0.0 {
//         QuadraticSolution::One(-b * rcp_2a)
//     } else {
//         let discriminant = discriminant.sqrt();
//         let p = (-b + discriminant) * rcp_2a;
//         let q = (-b - discriminant) * rcp_2a;
//         QuadraticSolution::Two(p.min(q), p.max(q))
//     }
// }

const MODE: Mode = Mode::Normal;

#[repr(align(16))]
struct Sphere {
    center: Vec3A, // center of the sphere
    radius: f32,   // radius of the sphere
    geom_id: u32,
    // geometry: Geometry<'a>,
}

#[repr(align(16))]
struct Instance<'a> {
    geometry: Geometry<'a>,
    scene: Scene<'a>,
    user_id: u32,
    local2world: Mat3A,
    world2local: Mat3A,
    normal2world: Mat3A,
    lower: Vec3A,
    upper: Vec3A,
}

fn create_analytical_sphere<'a>(
    device: &'a Device,
    scene: &'a mut Scene<'a>,
    center: Vec3A,
    radius: f32,
) -> Sphere {
    let mut geom = device.create_geometry(GeometryKind::USER).unwrap();
    geom.set_user_primitive_count(1);
    // geom.set_user_data(&mut sphere);
    geom.set_owned_user_data(Sphere {
        center,
        radius,
        geom_id: scene.attach_geometry(&geom),
        // geometry: geom.clone(),
    });
    geom.set_bounds_function(
        |bounds, prim_id, time_step, user_data: Option<&mut Sphere>| {
            let sphere = user_data.unwrap();
            let r = sphere.radius;
            let c = sphere.center;
            bounds.lower_x = sphere.center.x - sphere.radius;
            bounds.lower_y = sphere.center.y - sphere.radius;
            bounds.lower_z = sphere.center.z - sphere.radius;
            bounds.upper_x = sphere.center.x + sphere.radius;
            bounds.upper_y = sphere.center.y + sphere.radius;
            bounds.upper_z = sphere.center.z + sphere.radius;
        },
    );

    // if MODE == Mode::Normal {
    //     geom.set_intersect_function();
    //     geom.set_occluded_function();
    //     geom.set_intersect_filter_function();
    //     geom.set_occluded_filter_function();
    // } else {
    //     geom.set_intersect_function();
    //     geom.set_occluded_function();
    //     geom.set_intersect_filter_function();
    //     geom.set_occluded_filter_function();
    // }
    geom.commit();

    sphere
}

fn create_analytical_spheres<'a>(
    device: &'a Device,
    scene: &'a mut Scene<'a>,
    n: u32,
) -> AlignedVector<Sphere> {
    let mut geom = device.create_geometry(GeometryKind::USER).unwrap();
    let mut spheres = AlignedVector::<Sphere>::zeroed(n as usize, 16);
    let geom_id = scene.attach_geometry(&geom);
    spheres.iter_mut().for_each(|sphere| {
        sphere.geom_id = geom_id;
        // sphere.geometry = geom.clone();
    });
    geom.set_user_primitive_count(n);
    geom.set_user_data(&mut spheres);
    geom.set_bounds_function(
        |bounds, prim_id, time_step, user_data: Option<&mut Vec<Sphere>>| {
            let spheres = user_data.unwrap();
            let sphere = &spheres[prim_id as usize];
            bounds.lower_x = sphere.center.x - sphere.radius;
            bounds.lower_y = sphere.center.y - sphere.radius;
            bounds.lower_z = sphere.center.z - sphere.radius;
            bounds.upper_x = sphere.center.x + sphere.radius;
            bounds.upper_y = sphere.center.y + sphere.radius;
            bounds.upper_z = sphere.center.z + sphere.radius;
        },
    );
    // if MODE == Mode::Normal {
    //     geom.set_intersect_function();
    //     geom.set_occluded_function();
    //     geom.set_intersect_filter_function();
    //     geom.set_occluded_filter_function();
    // } else {
    //     geom.set_intersect_function();
    //     geom.set_occluded_function();
    //     geom.set_intersect_filter_function();
    //     geom.set_occluded_filter_function();
    // }
    geom.commit();
    spheres
}

fn sphere_intersect<'a>(
    ray_hit_n: RayHitN<'a>,
    valid_n: ValidityN<'a>,
    ctx: &mut IntersectContext,
    geom_id: u32,
    prim_id: u32,
    user_data: Option<&mut Sphere>,
) {
    assert_eq!(
        ray_hit_n.len(),
        1,
        "single ray sphere intersection but ray_hit_n.len() != 1"
    );
    if valid_n[0] != ValidMask::Valid {
        return;
    }
    let sphere = user_data.unwrap();
    let ray_n = ray_hit_n.ray_n();
    let ray_dir: Vec3A = ray_n.unit_dir(0).into();
    let ray_org: Vec3A = ray_n.org(0).into();
    let dir_o_c = ray_org - sphere.center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * ray_dir.dot(dir_o_c);
    let c = dir_o_c.dot(dir_o_c) - sphere.radius * sphere.radius;
    // let solution = solve_quadratic(a, b, c);
}

fn main() {
    let display = Display::new(
        DEFAULT_DISPLAY_WIDTH,
        DEFAULT_DISPLAY_HEIGHT,
        "triangle geometry",
    );
    let device = Device::new().unwrap();
    device.set_error_function(|err, msg| {
        println!("{}: {}", err, msg);
    });
    let scene = device.create_scene().unwrap();
}
