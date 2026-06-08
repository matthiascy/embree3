//! Scene-vs-scene collision (`rtcCollide`): embree reports broad-phase
//! candidate primitive pairs (their bounds need not actually overlap) between
//! two user-geometry scenes via a callback run on multiple threads.
mod common;

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        Arc, Mutex,
    },
    thread::ThreadId,
};

use embree3::{Bounds, Collision, Device, Geometry, GeometryKind, Scene};

/// A committed USER geometry with `n` unit boxes laid out along x so that each
/// overlaps its neighbours (box `i` spans `[i, i + 1.5]`). Enough primitives to
/// make embree's collider fan out across worker threads and batch its
/// callbacks.
fn user_boxes(device: &Device, n: u32) -> Geometry<'static> {
    let mut g = device.create_geometry(GeometryKind::USER).unwrap();
    g.set_primitive_count(n);
    g.set_bounds_function::<_, ()>(move |b: &mut Bounds, prim, _time, _user: Option<&()>| {
        let f = prim as f32;
        b.lower_x = f;
        b.lower_y = 0.0;
        b.lower_z = 0.0;
        b.upper_x = f + 1.5;
        b.upper_y = 1.0;
        b.upper_z = 1.0;
    });
    g.commit()
}

/// A committed single-primitive USER geometry whose only role is its AABB
/// (collision is broad-phase, so just the bounds function matters).
fn user_box(device: &Device, lo: [f32; 3], hi: [f32; 3]) -> Geometry<'static> {
    let mut g = device.create_geometry(GeometryKind::USER).unwrap();
    g.set_primitive_count(1);
    g.set_bounds_function::<_, ()>(move |b: &mut Bounds, _prim, _time, _user: Option<&()>| {
        b.lower_x = lo[0];
        b.lower_y = lo[1];
        b.lower_z = lo[2];
        b.upper_x = hi[0];
        b.upper_y = hi[1];
        b.upper_z = hi[2];
    });
    g.commit()
}

fn committed_scene(device: &Device, geom: &Geometry<'static>) -> Scene<'static> {
    let mut s = device.create_scene().unwrap();
    s.attach_geometry(geom);
    s.commit();
    s
}

#[test]
fn overlapping_user_boxes_report_a_collision_pair() {
    let device = common::device();
    let g0 = user_box(&device, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let g1 = user_box(&device, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]); // overlaps g0
    let s0 = committed_scene(&device, &g0);
    let s1 = committed_scene(&device, &g1);

    // The callback runs on embree's worker threads, so the captured state must be
    // Sync (Arc<Mutex<_>>); this also proves threaded user data is reached soundly.
    let pairs = Arc::new(Mutex::new(Vec::<(u32, u32, u32, u32)>::new()));
    let sink = pairs.clone();
    // SAFETY: both scenes are committed (`committed_scene`), from the same
    // `device`, and contain only single-time-step user geometries (`user_box`).
    unsafe {
        s0.collide(&s1, move |cs: &mut [Collision]| {
            let mut v = sink.lock().unwrap();
            v.extend(
                cs.iter()
                    .map(|c| (c.geomID0, c.primID0, c.geomID1, c.primID1)),
            );
        });
    }

    let v = pairs.lock().unwrap();
    assert!(
        !v.is_empty(),
        "overlapping user boxes must report at least one broad-phase pair"
    );
    assert!(
        v.iter()
            .any(|&(g0, p0, g1, p1)| g0 == 0 && p0 == 0 && g1 == 0 && p1 == 0),
        "the single prim of each scene should be reported as a candidate pair, got {v:?}"
    );
}

#[test]
fn broad_phase_is_conservative_user_does_narrow_phase() {
    // `rtcCollide` is a BROAD phase: with one primitive per scene it always emits
    // the single leaf pair as a *candidate*, even when the AABBs do not overlap
    // (it does no leaf-level cull). The user is responsible for the
    // narrow-phase test. This documents that contract: the disjoint boxes still
    // surface a candidate, which a real narrow-phase would reject.
    let device = common::device();
    let g0 = user_box(&device, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let g1 = user_box(&device, [10.0, 10.0, 10.0], [11.0, 11.0, 11.0]); // disjoint from g0
    let s0 = committed_scene(&device, &g0);
    let s1 = committed_scene(&device, &g1);

    let candidates = Arc::new(AtomicUsize::new(0));
    let c = candidates.clone();
    // SAFETY: committed, same device, single-time-step user geometries.
    unsafe {
        s0.collide(&s1, move |cs: &mut [Collision]| {
            c.fetch_add(cs.len(), Relaxed);
        });
    }

    assert!(
        candidates.load(Relaxed) >= 1,
        "broad phase should surface the candidate pair even for disjoint AABBs (the user \
         narrow-phases); embree reported none"
    );
}

#[test]
fn many_primitives_collide_soundly_under_concurrency() {
    // Thousands of overlapping primitives make embree fan the collision traversal
    // across worker threads and deliver collisions in batches. The atomic
    // counter and the `Mutex<HashSet>` of observed thread IDs must be updated
    // race-free; run under ThreadSanitizer to actually check that.
    let device = common::device();
    let g0 = user_boxes(&device, 4000);
    let g1 = user_boxes(&device, 4000);
    let s0 = committed_scene(&device, &g0);
    let s1 = committed_scene(&device, &g1);

    let count = Arc::new(AtomicUsize::new(0));
    let threads = Arc::new(Mutex::new(HashSet::<ThreadId>::new()));
    let (c, t) = (count.clone(), threads.clone());
    // SAFETY: committed, same device, single-time-step user geometries.
    unsafe {
        s0.collide(&s1, move |cs: &mut [Collision]| {
            c.fetch_add(cs.len(), Relaxed);
            t.lock().unwrap().insert(std::thread::current().id());
        });
    }

    assert!(
        count.load(Relaxed) > 0,
        "overlapping multi-primitive scenes must report candidate pairs"
    );
    // Informational (not asserted > 1: a low-core machine may use one thread).
    eprintln!(
        "collide ran callbacks on {} distinct thread(s)",
        threads.lock().unwrap().len()
    );
}

#[test]
fn self_collision_omits_identical_primitive_pairs() {
    // Self-collision: pass the same scene as both arguments. embree omits the
    // `(primitive, same primitive)` pair but does not guarantee ordering or
    // uniqueness (it may report both `(A, B)` and `(B, A)`), so we only assert that
    // no self-pair appears.
    let device = common::device();
    let g0 = user_box(&device, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let g1 = user_box(&device, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]); // overlaps g0
    let mut scene = device.create_scene().unwrap();
    scene.attach_geometry(&g0); // geomID 0
    scene.attach_geometry(&g1); // geomID 1
    scene.commit();

    let pairs = Arc::new(Mutex::new(Vec::<(u32, u32, u32, u32)>::new()));
    let sink = pairs.clone();
    // SAFETY: committed, one device, single-time-step user geometries, non-empty,
    // same scene (so identical BVH layout). The callback does not mutate the scene.
    unsafe {
        scene.collide(&scene, move |cs: &mut [Collision]| {
            let mut v = sink.lock().unwrap();
            v.extend(
                cs.iter()
                    .map(|c| (c.geomID0, c.primID0, c.geomID1, c.primID1)),
            );
        });
    }

    let v = pairs.lock().unwrap();
    assert!(
        !v.is_empty(),
        "two overlapping geometries in one scene should self-collide"
    );
    assert!(
        v.iter().all(|&(g0, p0, g1, p1)| !(g0 == g1 && p0 == p1)),
        "embree must omit (primitive, same primitive) self-pairs, got {v:?}"
    );
}
