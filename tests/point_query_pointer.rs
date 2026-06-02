//! Proves Scene::point_query passes a usable pointer to its callback.
//! `scene.rs` currently hands embree `point_query_user_data.data` while the
//! trampoline casts userPtr back to `*mut PointQueryUserData` (type confusion).

mod common;

use std::{cell::RefCell, rc::Rc};

use embree::{PointQuery, PointQueryContext, INVALID_ID};

#[test]
fn point_query_invokes_callback_with_live_state() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let mut tri = common::unit_triangle(&device);
    tri.commit();
    scene.attach_geometry(&tri);
    scene.commit();

    let ran = Rc::new(RefCell::new(false));
    let ran_in_cb = ran.clone();

    let mut query = PointQuery {
        x: 0.25,
        y: 0.25,
        z: 0.0,
        time: 0.0,
        radius: 1.0,
    };
    // `RTCPointQueryContext` derives no `Default`, and the embree initializer
    // (`rtcInitPointQueryContext`) is not wrapped. Construct it explicitly: an
    // empty instance stack with the no-instance sentinel. The transform
    // matrices are only read when instances are pushed, so zeros are fine for
    // this non-instanced scene.
    let mut ctx = PointQueryContext {
        world2inst: [[0.0; 16]; 1],
        inst2world: [[0.0; 16]; 1],
        instID: [INVALID_ID; 1],
        instStackSize: 0,
    };

    common::clobber_stack();
    scene.point_query::<_, ()>(
        &mut query,
        &mut ctx,
        Some(
            |_q: &mut PointQuery,
             _c: &mut PointQueryContext,
             _d: Option<&mut ()>,
             _prim,
             _geom,
             _s| {
                *ran_in_cb.borrow_mut() = true;
                false
            },
        ),
        None,
    );

    assert!(
        *ran.borrow(),
        "point query callback must be invoked with live captured state"
    );
}
