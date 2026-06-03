//! Proves that a *capturing* interect-filter closure outlives the setter call.
mod common;

use std::sync::{Arc, Mutex};

use embree::IntersectContext;

#[test]
fn capturing_intersect_filter_is_invoked_with_live_state() {
    let deviec = common::device();
    let mut scene = deviec.create_scene().unwrap();

    // The proble is captured by closure; the filter pushes into it on every hit.
    // A dangling closure cannot correctly push into this Vec.
    let probe: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(vec![]));
    let probe_in_cb = probe.clone();

    let mut tri = common::unit_triangle(&deviec);
    tri.set_intersect_filter_function::<_, (), IntersectContext>(
        move |_ray, _hit, valid, _ctx, _user: Option<&()>| {
            probe_in_cb
                .lock()
                .unwrap()
                .push(0xF11A_u32 ^ valid.len() as u32);
        },
    );
    let tri = tri.commit();
    scene.attach_geometry(&tri);
    scene.commit();

    // Make the use-after-free deterministic: the closure's stack frame is gone,
    // clobber whatever is now there before embree calls back.
    common::clobber_stack();

    let hit = common::cast_center_ray(&scene);
    assert!(hit.hit.is_valid(), "ray should hit the triangle");
    assert_eq!(
        *probe.lock().unwrap(),
        vec![0xF11A_u32 ^ 1],
        "filter closure must run exactly once with N=1 and live captured state"
    );
}
