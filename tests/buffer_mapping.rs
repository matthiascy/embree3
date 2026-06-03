//! Demonstrates the ability to *map* an already-bound geometry-local buffer
//! back to a typed slice, for three phases of a geometry's life:
//!
//! * build phase, sole owner: [`GeometryBuilder::map_buffer_mut`] /
//!   [`GeometryBuilder::map_buffer`] (re-fill / inspect before `commit`);
//! * committed, read-only: [`Geometry::map_buffer`];
//! * attached, write through `&mut Scene`: [`Scene::with_geometry_buffer_mut`]
//!   (the borrow checker proves no concurrent traversal; the slice cannot
//!   escape the closure).
//!
//! The load-bearing proof is `dynamic_edit_through_scene_moves_geometry`: the
//! mapped slice must be embree's *live* buffer, so writing through it and
//! re-committing actually changes what rays hit. A mapping that handed back a
//! copy, or the wrong pointer, would silently fail this; a rendered image
//! could not.
mod common;

use embree::{BufferUsage, Error};

/// Build phase, sole owner: re-fill a `set_new_buffer` allocation through
/// `map_buffer_mut`, then read it back through `map_buffer`. Proves both map to
/// the *same* live allocation (write-then-read round-trips).
#[test]
fn builder_map_refill_round_trips() {
    let device = common::device();
    let mut tri = common::unit_triangle(&device);

    let moved = [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]];
    tri.map_buffer_mut::<[f32; 3]>(BufferUsage::VERTEX, 0)
        .unwrap()
        .copy_from_slice(&moved);

    let view = tri.map_buffer::<[f32; 3]>(BufferUsage::VERTEX, 0).unwrap();
    assert_eq!(
        view.as_ref(),
        &moved,
        "map_buffer must read back exactly what map_buffer_mut wrote"
    );
}

/// A build-phase write survives `commit`: the committed `Geometry`'s read map
/// observes the bytes set on the builder. Proves the live buffer is shared
/// across the typestate transition (not reallocated/copied by `commit`).
#[test]
fn committed_read_map_sees_build_phase_writes() {
    let device = common::device();
    let mut tri = common::unit_triangle(&device);
    let moved = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
    tri.map_buffer_mut::<[f32; 3]>(BufferUsage::VERTEX, 0)
        .unwrap()
        .copy_from_slice(&moved);

    let geom = tri.commit();
    let view = geom.map_buffer::<[f32; 3]>(BufferUsage::VERTEX, 0).unwrap();
    assert_eq!(
        view.as_ref(),
        &moved,
        "the committed geometry must expose the same buffer the builder filled"
    );
}

/// Attach a unit triangle, confirm the center ray
/// hits it. Then translate every vertex +10 in x *through the mapped slice*,
/// re-commit, and confirm the ray now misses; translate back and confirm it
/// hits again. This can only pass if `with_geometry_buffer_mut` hands back
/// embree's live vertex buffer (a copy would never change traversal).
#[test]
fn dynamic_edit_through_scene_moves_geometry() {
    let device = common::device();
    let mut scene = device.create_scene().unwrap();
    let tri = common::unit_triangle(&device).commit();
    let id = scene.attach_geometry(&tri);
    scene.commit();
    assert!(
        common::cast_center_ray(&scene).hit.is_valid(),
        "the triangle should be hit at its original position"
    );

    // Translate the triangle far off the ray's path, through the mapped slice.
    let written = scene
        .with_geometry_buffer_mut::<[f32; 3], _>(id, BufferUsage::VERTEX, 0, |verts| {
            for v in verts.iter_mut() {
                v[0] += 10.0;
            }
            verts.len()
        })
        .unwrap();
    assert_eq!(written, 3, "the closure should see all 3 mapped vertices");
    scene.update_geometry_buffer(id, BufferUsage::VERTEX, 0);
    scene.commit_geometry(id);
    scene.commit();
    assert!(
        !common::cast_center_ray(&scene).hit.is_valid(),
        "after moving the triangle off-axis the ray must miss, proving the mapped write reached \
         embree's live buffer"
    );

    // Move it back; the ray hits again.
    scene
        .with_geometry_buffer_mut::<[f32; 3], _>(id, BufferUsage::VERTEX, 0, |verts| {
            for v in verts.iter_mut() {
                v[0] -= 10.0;
            }
        })
        .unwrap();
    scene.update_geometry_buffer(id, BufferUsage::VERTEX, 0);
    scene.commit_geometry(id);
    scene.commit();
    assert!(
        common::cast_center_ray(&scene).hit.is_valid(),
        "restoring the vertices should make the triangle hittable again"
    );
}

/// Mapping rejects unbound slots, non-tiling element types, and (for the scene
/// path) unattached ids; the `map_local` layout checks, surfaced as
/// `INVALID_ARGUMENT` rather than handing back an unsound slice.
#[test]
fn map_rejects_invalid_requests() {
    let device = common::device();
    let geom = common::unit_triangle(&device).commit();

    // Nothing bound at slot 5.
    assert_eq!(
        geom.map_buffer::<[f32; 3]>(BufferUsage::VERTEX, 5).err(),
        Some(Error::INVALID_ARGUMENT),
        "an unbound slot must not map"
    );

    // The 3-vertex VERTEX buffer is 36 bytes; `[f32; 4]` (16 bytes) does not
    // tile it (36 % 16 != 0), so the element type is rejected.
    assert_eq!(
        geom.map_buffer::<[f32; 4]>(BufferUsage::VERTEX, 0).err(),
        Some(Error::INVALID_ARGUMENT),
        "an element type that does not tile the allocation must be rejected"
    );

    // The scene path rejects a missing geometry id.
    let mut scene = device.create_scene().unwrap();
    let r = scene.with_geometry_buffer_mut::<[f32; 3], _>(999, BufferUsage::VERTEX, 0, |_verts| {
        unreachable!("closure must not run for an unattached id")
    });
    assert_eq!(
        r.err(),
        Some(Error::INVALID_ARGUMENT),
        "mapping an unattached geometry id must fail"
    );
}
