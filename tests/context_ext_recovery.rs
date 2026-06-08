//! A callback recovers the typed per-ray extension from the base
//! `IntersectContext` via the (unsafe) `ext` accessor, and sees the exact data
//! the query attached. This proves the "base context + localized-unsafe
//! recovery" design: the callback type no longer carries the context type, yet
//! the recovered `&mut T` aliases the query's own `IntersectContextExt<T>.ext`.
mod common;

use embree3::{IntersectContextExt, Ray, RayHit, SoAHit};

#[derive(Default)]
struct Probe {
    calls: u32,
    seen_geom: u32,
}

#[test]
fn callback_recovers_typed_context_ext() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let mut tri = common::unit_triangle(&device);

    // The geometry intersect filter runs automatically for built-in geometry.
    // The callback takes the *base* `&mut IntersectContext` and recovers the
    // typed extension explicitly.
    tri.set_intersect_filter_function::<_, ()>(
        move |_ray, hit, _valid, ctx, _user: Option<&()>| {
            // SAFETY: the query below uses `IntersectContextExt<Probe>`.
            let probe = unsafe { ctx.ext_mut::<Probe>() };
            probe.calls += 1;
            probe.seen_geom = hit.geom_id(0);
        },
    );

    let tri = tri.commit();
    scene.attach_geometry(&tri); // geomID 0
    scene.commit();

    let mut ctx = IntersectContextExt::coherent(Probe::default());
    let mut rh = RayHit::from(Ray::segment(
        [0.25, 0.25, -1.0],
        [0.0, 0.0, 1.0],
        0.0,
        f32::INFINITY,
    ));
    scene.intersect(&mut ctx, &mut rh);

    assert!(
        ctx.ext.calls >= 1,
        "the filter ran and recovered the typed extension"
    );
    assert_eq!(
        ctx.ext.seen_geom, 0,
        "the recovered ext is the query's own data (saw the hit's geomID)"
    );
}
