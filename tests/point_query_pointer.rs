//! Proves `Scene::point_query` invokes its callback with live captured state.
//! The single query takes no `data` parameter (the closure captures its own
//! accumulator) and carries no `Send`/`Sync`/`'static` bound, so a non-`Send`
//! `Rc<RefCell<_>>` capture is accepted and observed after the call.

mod common;

use std::{cell::RefCell, rc::Rc};

use embree3::{PointQuery, PointQueryContext};

#[test]
fn point_query_invokes_callback_with_live_state() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
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
    let mut ctx = PointQueryContext::new();

    common::clobber_stack();
    scene.point_query(&mut query, &mut ctx, |_q, _c, _prim, _geom, _s| {
        *ran_in_cb.borrow_mut() = true;
        false
    });

    assert!(
        *ran.borrow(),
        "point query callback must be invoked with live captured state"
    );
}
