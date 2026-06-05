mod common;

use embree::{CbKind, IntersectContext, Ray};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[test]
fn each_callback_sees_its_own_data() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let mut tri = common::unit_triangle(&device);

    let from_intersect = Arc::new(Mutex::new(None::<i32>));
    let from_occluded = Arc::new(Mutex::new(None::<i32>));
    let (ci, co) = (from_intersect.clone(), from_occluded.clone());

    tri.set_intersect_filter_function_owned::<_, i32, IntersectContext>(
        move |_r, _h, _v, _c, ud| {
            *ci.lock().unwrap() = ud.copied();
        },
        7i32,
    );
    tri.set_occluded_filter_function_owned::<_, i32, IntersectContext>(
        move |_r, _h, _v, _c, ud| {
            *co.lock().unwrap() = ud.copied();
        },
        99i32,
    );
    let geom = tri.commit();
    scene.attach_geometry(&geom);
    scene.commit();

    let _ = common::cast_center_ray(&scene);
    let mut ctx = IntersectContext::coherent();
    let mut shadow = Ray::segment([0.25, 0.25, -1.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY);
    let _ = scene.occluded(&mut ctx, &mut shadow);

    assert_eq!(
        *from_intersect.lock().unwrap(),
        Some(7),
        "intersect filter sees its own data"
    );
    assert_eq!(
        *from_occluded.lock().unwrap(),
        Some(99),
        "occluded filter sees its own, different data"
    );
}

#[test]
fn replacing_callback_swaps_closure_and_its_own_data() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let mut tri = common::unit_triangle(&device);

    // A's capture; freed when A's slot is replaced.
    let a_alive = Arc::new(());
    let a_weak = Arc::downgrade(&a_alive);
    let seen = Arc::new(Mutex::new(String::new()));
    let (sa, sb) = (seen.clone(), seen.clone());

    tri.set_intersect_filter_function_owned::<_, i32, IntersectContext>(
        move |_r, _h, _v, _c, _ud| {
            let _hold = &a_alive;
            *sa.lock().unwrap() = "A".into();
        },
        7i32,
    );
    tri.set_intersect_filter_function_owned::<_, f32, IntersectContext>(
        move |_r, _h, _v, _c, ud| {
            *sb.lock().unwrap() = format!("B:{:?}", ud.copied());
        },
        2.5f32,
    );

    assert!(
        a_weak.upgrade().is_none(),
        "A's closure and its owned i32 are freed on replacement"
    );

    let geom = tri.commit();
    scene.attach_geometry(&geom);
    scene.commit();
    let _ = common::cast_center_ray(&scene);
    assert_eq!(
        *seen.lock().unwrap(),
        "B:Some(2.5)",
        "B runs and sees its own f32 - per-callback, no resolution"
    );
}

#[test]
fn owned_data_dropped_once_and_readable_via_getter() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Witness(u8);
    impl Drop for Witness {
        fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); }
    }

    let device = common::device();
    let mut tri = common::unit_triangle(&device);

    tri.set_intersect_filter_function_owned::<_, Witness, IntersectContext>(
        |_r, _h, _v, _c, _ud| {},
        Witness(1),
    );
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
    assert_eq!(
        tri.callback_data::<Witness>(CbKind::IntersectFilter)
            .map(|w| w.0),
        Some(1),
    );

    // replaces -> Witness(1) dropped once
    tri.set_intersect_filter_function_owned::<_, Witness, IntersectContext>(
        |_r, _h, _v, _c, _ud| {},
        Witness(2),
    );
    assert_eq!(
        DROPS.load(Ordering::SeqCst),
        1,
        "old owned data dropped exactly once on replace"
    );
    drop(tri.commit()); // Witness(2) dropped on geometry drop
    assert_eq!(DROPS.load(Ordering::SeqCst), 2);
}

#[test]
fn borrowed_data_reaches_callback_zero_copy() {
    use embree::{BufferUsage, Format, GeometryKind};

    // Non-'static data, declared BEFORE the builder so it outlives it. The
    // borrow is refcount-free, no `Arc`, no allocation beyond `table` itself.
    let table: Vec<u32> = vec![10, 20, 30];

    let device = common::device();
    let mut scene = device.create_scene().unwrap();

    // Built inline (not `common::unit_triangle`, which is
    // `GeometryBuilder<'static>` and would force `&'static table`) so the
    // geometry's `'buf` is free to be tied to `table`'s lifetime.
    let mut tri = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    tri.set_new_buffer::<[f32; 3]>(BufferUsage::VERTEX, 0, Format::FLOAT3, 12, 3)
        .unwrap()
        .copy_from_slice(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    tri.set_new_buffer::<[u32; 3]>(BufferUsage::INDEX, 0, Format::UINT3, 12, 1)
        .unwrap()
        .copy_from_slice(&[[0, 1, 2]]);

    let seen = Arc::new(Mutex::new(None::<u32>));
    let sc = seen.clone();
    tri.set_intersect_filter_function_borrowed::<_, Vec<u32>, IntersectContext>(
        move |_r, _h, _v, _c, ud| {
            *sc.lock().unwrap() = ud.map(|t| t.iter().copied().sum::<u32>());
        },
        &table,
    );

    let geom = tri.commit();
    scene.attach_geometry(&geom);
    scene.commit();
    let _ = common::cast_center_ray(&scene);

    assert_eq!(
        *seen.lock().unwrap(),
        Some(60),
        "the filter reads the borrowed, non-'static, refcount-free table"
    );
}
