//! Packet point queries (`rtcPointQuery4/8/16`). Each lane is an independent
//! query with its own per-lane accumulator, routed through the per-lane
//! `userPtr`. This proves per-lane data routing (each lane's callback receives
//! *its own* query and writes *its own* accumulator) and that the valid mask is
//! honoured (inactive lanes are never invoked).
mod common;

use embree3::{PointQuery, PointQuery4, PointQueryContext, ValidMask, ValidMaskN};

#[test]
fn point_query_captures_state_and_returns_changed() {
    // The single query takes no `data` param: the closure captures its own
    // accumulator, and the method returns embree's `bool` (was the query changed).
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    scene.attach_geometry(&tri);
    scene.commit();

    let mut hits = 0u32; // captured accumulator, no `data` param needed
    let mut query = PointQuery {
        x: 0.25,
        y: 0.25,
        z: 0.0,
        time: 0.0,
        radius: 1.0,
    };
    let mut ctx = PointQueryContext::new();

    let changed = scene.point_query(&mut query, &mut ctx, |q, _ctx, _prim, _geom, _scale| {
        hits += 1;
        q.radius = 0.0; // shrink: report a change
        true
    });

    assert!(hits > 0, "callback must run and mutate captured state");
    assert!(
        changed,
        "point_query must return true when the callback reports a change"
    );
}

#[test]
fn point_query4_routes_per_lane_data_and_honours_valid() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    scene.attach_geometry(&tri);
    scene.commit();

    // The triangle spans [0,1] x [0,1] in the z = 0 plane. Lanes 0 and 2 sit on
    // it (callback fires); lane 3 is disabled via the valid mask. Each lane's
    // query x is distinct, so we can prove routing by having every invocation
    // record the query's own x into that lane's accumulator.
    let mut queries = PointQuery4 {
        x: [0.25, 100.0, 0.5, 0.25],
        y: [0.25, 100.0, 0.5, 0.25],
        z: [0.0, 0.0, 0.0, 0.0],
        time: [0.0; 4],
        radius: [1.0, 1.0, 1.0, 1.0],
    };
    let mut valid = ValidMaskN::all_active();
    valid.lanes_mut()[3] = ValidMask::Invalid; // lane 3 disabled

    let mut ctx = PointQueryContext::new();
    const SENTINEL: f32 = -42.0;
    let mut seen_x: [f32; 4] = [SENTINEL; 4];

    let changed = scene.point_query4(
        &valid,
        &mut queries,
        &mut ctx,
        |q: &mut PointQuery, _c: &mut PointQueryContext, slot: &mut f32, _prim, _geom, _scale| {
            *slot = q.x; // record THIS lane's own query into THIS lane's accumulator
            false // do not shrink the query radius
        },
        &mut seen_x,
    );

    // No lane reported a change, so the embree `bool` propagates out as `false`.
    assert!(
        !changed,
        "no callback reported a change, so the result must be false"
    );

    // Active lanes on the triangle each received their OWN query, not a
    // neighbour's.
    assert_eq!(
        seen_x[0], 0.25,
        "lane 0 must receive its own query (x=0.25)"
    );
    assert_eq!(seen_x[2], 0.5, "lane 2 must receive its own query (x=0.5)");
    // Inactive lane is never touched.
    assert_eq!(seen_x[3], SENTINEL, "inactive lane 3 must never be invoked");
    // The far lane may or may not be visited (conservative bbox culling); if it
    // is, it must only ever see its OWN query, never a neighbour's.
    assert!(
        seen_x[1] == SENTINEL || seen_x[1] == 100.0,
        "lane 1 must only ever see its own query, got {}",
        seen_x[1]
    );
}

#[test]
fn point_query4_returns_true_when_a_lane_reports_a_change() {
    // Only lane 0 sits on the triangle; its callback shrinks the radius and
    // returns `true`. The packet's embree `bool` result must propagate that out.
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    scene.attach_geometry(&tri);
    scene.commit();

    let mut queries = PointQuery4 {
        x: [0.25, 100.0, 100.0, 100.0],
        y: [0.25, 100.0, 100.0, 100.0],
        z: [0.0; 4],
        time: [0.0; 4],
        radius: [1.0; 4],
    };
    let valid = ValidMaskN::all_active();
    let mut ctx = PointQueryContext::new();
    let mut data = [(); 4]; // no per-lane state needed (D = ())

    let changed = scene.point_query4(
        &valid,
        &mut queries,
        &mut ctx,
        |q: &mut PointQuery, _c: &mut PointQueryContext, _d: &mut (), _prim, _geom, _scale| {
            q.radius = 0.0; // shrink: report a change
            true
        },
        &mut data,
    );

    assert!(
        changed,
        "point_query4 must return true when a lane reports a change"
    );
}
