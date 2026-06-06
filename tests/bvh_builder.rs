//! Intergration tests for the standalone BVH builder (`Bvh::build_scoped`).

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    thread::ThreadId,
};

use embree3::{
    Allocator, Bounds, BuildConfig, BuildPrimitive, BuildQuality, BvhBuilder, BvhNode, BvhResult,
    ChildBounds, Children, Device, NodePtr,
};

/// Placeholder bounds written at `create_node` and overwritten by `set_bounds`.
const EMPTY_BOUNDS: Bounds = Bounds {
    lower_x: f32::INFINITY,
    lower_y: f32::INFINITY,
    lower_z: f32::INFINITY,
    align0: 0.0,
    upper_x: f32::NEG_INFINITY,
    upper_y: f32::NEG_INFINITY,
    upper_z: f32::NEG_INFINITY,
    align1: 0.0,
};

/// Branching-2 node. `kids` are `Option` because `create_node` runs before the
/// children exist; `set_children` fills them. `Option<NodePtr>` is 8 bytes.
#[derive(Clone, Copy)]
enum Node<'id> {
    Inner {
        bounds: [Bounds; 2],
        kids: [Option<NodePtr<'id, Node<'id>>>; 2],
    },
    Leaf {
        prim_count: u32,
    },
}
unsafe impl<'id> BvhNode for Node<'id> {}

/// A builder that records what embree did: node/leaf counts, the primIDs handed
/// to `create_leaf`, the distinct callback threads, and progress calls. It has
/// `SPATIAL_SPLITS` and `PROGRESS` enabled so the same type drives every test
/// (splits only actually fire at HIGH quality; progress is harmless elsewhere).
#[derive(Default)]
struct Recorder {
    nodes: AtomicUsize,
    leaves: AtomicUsize,
    prim_ids: Mutex<Vec<u32>>,
    threads: Mutex<HashSet<ThreadId>>,
    progress_calls: AtomicUsize,
    split_calls: AtomicUsize,
}

impl BvhBuilder for Recorder {
    type Node<'id> = Node<'id>;
    const MAX_CHILDREN: usize = 2;
    const SPATIAL_SPLITS: bool = true;
    const PROGRESS: bool = true;

    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Node<'id> {
        self.nodes.fetch_add(1, Ordering::Relaxed);
        self.threads
            .lock()
            .unwrap()
            .insert(std::thread::current().id());
        a.alloc(Node::Inner {
            bounds: [EMPTY_BOUNDS; 2],
            kids: [None; 2],
        })
    }

    fn set_children<'id>(&self, node: &mut Node<'id>, children: Children<'id, Node<'id>>) {
        if let Node::Inner { kids, .. } = node {
            for i in 0..children.len().min(2) {
                kids[i] = children.get(i);
            }
        }
    }

    fn set_bounds<'id>(&self, node: &mut Node<'id>, bounds: ChildBounds<'_>) {
        if let Node::Inner { bounds: b, .. } = node {
            for i in 0..bounds.len().min(2) {
                if let Some(cb) = bounds.get(i) {
                    b[i] = *cb;
                }
            }
        }
    }

    fn create_leaf<'id>(&self, a: &Allocator<'id>, prims: &[BuildPrimitive]) -> &'id mut Node<'id> {
        self.leaves.fetch_add(1, Ordering::Relaxed);
        self.threads
            .lock()
            .unwrap()
            .insert(std::thread::current().id());
        let mut ids = self.prim_ids.lock().unwrap();
        ids.extend(prims.iter().map(|p| p.primID));
        a.alloc(Node::Leaf {
            prim_count: prims.len() as u32,
        })
    }

    // Overrides the default split only to count invocations; the geometry is the
    // same conservative geometric AABB split the default performs.
    fn split(&self, prim: &BuildPrimitive, dim: u32, pos: f32) -> (Bounds, Bounds) {
        self.split_calls.fetch_add(1, Ordering::Relaxed);
        let base = Bounds {
            lower_x: prim.lower_x,
            lower_y: prim.lower_y,
            lower_z: prim.lower_z,
            align0: 0.0,
            upper_x: prim.upper_x,
            upper_y: prim.upper_y,
            upper_z: prim.upper_z,
            align1: 0.0,
        };
        let mut lo = base;
        let mut hi = base;
        match dim {
            0 => {
                lo.upper_x = pos;
                hi.lower_x = pos;
            }
            1 => {
                lo.upper_y = pos;
                hi.lower_y = pos;
            }
            _ => {
                lo.upper_z = pos;
                hi.lower_z = pos;
            }
        }
        (lo, hi)
    }

    fn progress(&self, _fraction: f64) { self.progress_calls.fetch_add(1, Ordering::Relaxed); }
}

/// `n` unit boxes laid out along x: box `i` spans `[i, i+1] x [0,1] x [0,1]`,
/// `geomID = 0`, `primID = i`.
fn make_prims(n: u32) -> Vec<BuildPrimitive> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            BuildPrimitive {
                lower_x: f,
                lower_y: 0.0,
                lower_z: 0.0,
                geomID: 0,
                upper_x: f + 1.0,
                upper_y: 1.0,
                upper_z: 1.0,
                primID: i,
            }
        })
        .collect()
}

/// Sum the primitives across all leaves by walking via `resolve`.
fn sum_prims<'id>(r: &BvhResult<'id, Recorder>, n: &Node<'id>) -> u32 {
    match n {
        Node::Leaf { prim_count } => *prim_count,
        Node::Inner { kids, .. } => kids
            .iter()
            .flatten()
            .map(|k| sum_prims(r, r.resolve(*k)))
            .sum(),
    }
}

#[test]
fn build_covers_all_primitives_via_navigation() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims = make_prims(64);
    let cfg = BuildConfig::default(); // MEDIUM, no splits
    let recorder = Recorder::default();

    let covered = bvh
        .build_scoped(&cfg, &mut prims, &recorder, |r| {
            r.root().map(|root| sum_prims(&r, root)).unwrap_or(0)
        })
        .unwrap();

    assert_eq!(covered, 64, "every primitive must reach a leaf");
}

#[test]
fn create_leaf_sees_every_primitive_id_once_at_medium() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims = make_prims(64);
    let cfg = BuildConfig::default(); // MEDIUM: no spatial splits, so no primID duplication
    let recorder = Recorder::default();

    bvh.build_scoped(&cfg, &mut prims, &recorder, |_r| ())
        .unwrap();

    let mut ids = recorder.prim_ids.into_inner().unwrap();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..64).collect::<Vec<u32>>(),
        "create_leaf must see every primID exactly once"
    );
}

#[test]
fn shared_user_data_is_reached_soundly() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims = make_prims(4096); // large enough to fan out across threads
    let cfg = BuildConfig::default();
    let recorder = Recorder::default();

    let covered = bvh
        .build_scoped(&cfg, &mut prims, &recorder, |r| {
            r.root().map(|root| sum_prims(&r, root)).unwrap_or(0)
        })
        .unwrap();

    // Wiring + sound shared mutation under embree's threading.
    assert_eq!(covered, 4096);
    assert!(recorder.nodes.load(Ordering::Relaxed) > 0);
    assert!(recorder.leaves.load(Ordering::Relaxed) > 0);

    // Informational only: how many distinct threads ran callbacks. We do NOT
    // assert > 1, since a low-core machine or a small build may use one thread.
    let n_threads = recorder.threads.lock().unwrap().len();
    eprintln!("bvh build ran callbacks on {n_threads} distinct thread(s)");
    assert!(n_threads >= 1);
}

#[test]
fn root_child_bounds_enclose_all_inputs() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    // 64 prims with default max_leaf_size (32) forces an Inner root.
    let mut prims = make_prims(64);
    let cfg = BuildConfig::default();
    let recorder = Recorder::default();

    // Union of the root's child bounds, computed inside the scope.
    let (lo, hi) = bvh
        .build_scoped(&cfg, &mut prims, &recorder, |r| {
            match r.root().expect("non-empty build has a root") {
                Node::Inner { bounds, .. } => {
                    let mut lo = [f32::INFINITY; 3];
                    let mut hi = [f32::NEG_INFINITY; 3];
                    for b in bounds {
                        lo[0] = lo[0].min(b.lower_x);
                        lo[1] = lo[1].min(b.lower_y);
                        lo[2] = lo[2].min(b.lower_z);
                        hi[0] = hi[0].max(b.upper_x);
                        hi[1] = hi[1].max(b.upper_y);
                        hi[2] = hi[2].max(b.upper_z);
                    }
                    (lo, hi)
                }
                Node::Leaf { .. } => panic!("expected an inner root for 64 primitives"),
            }
        })
        .unwrap();

    // Input union is x in [0, 64], y in [0, 1], z in [0, 1].
    assert!(
        lo[0] <= 0.0 && hi[0] >= 64.0,
        "x not enclosed: {lo:?}..{hi:?}"
    );
    assert!(
        lo[1] <= 0.0 && hi[1] >= 1.0,
        "y not enclosed: {lo:?}..{hi:?}"
    );
    assert!(
        lo[2] <= 0.0 && hi[2] >= 1.0,
        "z not enclosed: {lo:?}..{hi:?}"
    );
}

#[test]
fn empty_input_yields_none_root() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims: Vec<BuildPrimitive> = Vec::new();
    let cfg = BuildConfig::default();
    let recorder = Recorder::default();

    let is_none = bvh
        .build_scoped(&cfg, &mut prims, &recorder, |r| r.root().is_none())
        .unwrap();

    assert!(is_none, "empty input must yield root() == None");
    // The short-circuit means embree was never called, so no leaf was created.
    assert_eq!(recorder.leaves.load(Ordering::Relaxed), 0);
}

#[test]
fn same_bvh_rebuilds_sequentially() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let cfg = BuildConfig::default();
    let recorder = Recorder::default();

    // 1) non-empty
    let mut p1 = make_prims(32);
    let c1 = bvh
        .build_scoped(&cfg, &mut p1, &recorder, |r| {
            r.root().map(|n| sum_prims(&r, n)).unwrap_or(0)
        })
        .unwrap();
    assert_eq!(c1, 32);

    // 2) empty on the SAME bvh (deferred arena reset path)
    let mut p2: Vec<BuildPrimitive> = Vec::new();
    let empty = bvh
        .build_scoped(&cfg, &mut p2, &recorder, |r| r.root().is_none())
        .unwrap();
    assert!(empty, "empty rebuild must yield None");

    // 3) non-empty again on the SAME bvh; must build correctly after the empty
    //    rebuild
    let mut p3 = make_prims(48);
    let c3 = bvh
        .build_scoped(&cfg, &mut p3, &recorder, |r| {
            r.root().map(|n| sum_prims(&r, n)).unwrap_or(0)
        })
        .unwrap();
    assert_eq!(
        c3, 48,
        "the same Bvh must rebuild correctly after an empty rebuild"
    );
}

#[test]
fn high_quality_with_spatial_splits_still_covers_all_primitives() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    // Large, heavily overlapping prims maximize the SAH overlap cost so the spatial
    // splitter has strong incentive to split: 16-wide boxes over a 16x16 unit grid,
    // each overlapping ~16 neighbours.
    let mut prims: Vec<BuildPrimitive> = (0..256)
        .map(|i| BuildPrimitive {
            lower_x: (i % 16) as f32,
            lower_y: (i / 16) as f32,
            lower_z: 0.0,
            geomID: 0,
            upper_x: (i % 16) as f32 + 16.0,
            upper_y: (i / 16) as f32 + 16.0,
            upper_z: 1.0,
            primID: i,
        })
        .collect();

    let cfg = BuildConfig {
        quality: BuildQuality::HIGH,
        ..Default::default()
    };
    let recorder = Recorder::default();

    // Recorder has SPATIAL_SPLITS = true, so HIGH may trigger splitPrimitive and
    // the 2x capacity reserve. The leaf primID SET (deduped) must always cover
    // every input; counts may exceed 256 because a split primitive lands in
    // multiple leaves.
    bvh.build_scoped(&cfg, &mut prims, &recorder, |_r| ())
        .unwrap();

    let ids: HashSet<u32> = recorder.prim_ids.lock().unwrap().iter().copied().collect();
    let expected: HashSet<u32> = (0..256).collect();
    assert_eq!(
        ids, expected,
        "spatial-split build must still reference every primID"
    );

    // EMPIRICAL (embree 3.13.5, the version this crate gates on): embree documents
    // that a split callback makes spatial splitting *possible*, not that every
    // HIGH build calls it. This dataset (large overlapping prims) reliably
    // forces splits in 3.13.5, so we assert the enabled split path actually
    // executes. If a future embree changes the heuristic and this fails, adjust
    // the dataset rather than dropping the check.
    assert!(
        recorder.split_calls.load(Ordering::Relaxed) > 0,
        "this overlapping dataset is expected to force spatial splits at HIGH on embree 3.13.5"
    );
}

#[test]
fn progress_callback_is_invoked_when_enabled() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims = make_prims(1024);
    let cfg = BuildConfig::default();
    let recorder = Recorder::default(); // PROGRESS = true

    bvh.build_scoped(&cfg, &mut prims, &recorder, |_r| ())
        .unwrap();

    // Report-only: embree 3.13.5 ignores the return, but it DOES call the monitor.
    assert!(
        recorder.progress_calls.load(Ordering::Relaxed) > 0,
        "progress() must be invoked when PROGRESS == true"
    );
}

/// Same node layout as `Recorder` but with both hook consts left at their
/// `false` defaults, so embree should register neither the split nor the
/// progress trampoline. The counters detect any (incorrect) invocation.
#[derive(Default)]
struct Disabled {
    split_calls: AtomicUsize,
    progress_calls: AtomicUsize,
}

impl BvhBuilder for Disabled {
    type Node<'id> = Node<'id>;
    const MAX_CHILDREN: usize = 2;
    // SPATIAL_SPLITS and PROGRESS default to false.

    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Node<'id> {
        a.alloc(Node::Inner {
            bounds: [EMPTY_BOUNDS; 2],
            kids: [None; 2],
        })
    }
    fn set_children<'id>(&self, node: &mut Node<'id>, c: Children<'id, Node<'id>>) {
        if let Node::Inner { kids, .. } = node {
            for i in 0..c.len().min(2) {
                kids[i] = c.get(i);
            }
        }
    }
    fn set_bounds<'id>(&self, node: &mut Node<'id>, bnds: ChildBounds<'_>) {
        if let Node::Inner { bounds, .. } = node {
            for i in 0..bnds.len().min(2) {
                if let Some(cb) = bnds.get(i) {
                    bounds[i] = *cb;
                }
            }
        }
    }
    fn create_leaf<'id>(&self, a: &Allocator<'id>, prims: &[BuildPrimitive]) -> &'id mut Node<'id> {
        a.alloc(Node::Leaf {
            prim_count: prims.len() as u32,
        })
    }
    fn split(&self, _p: &BuildPrimitive, _d: u32, _pos: f32) -> (Bounds, Bounds) {
        self.split_calls.fetch_add(1, Ordering::Relaxed);
        (EMPTY_BOUNDS, EMPTY_BOUNDS)
    }
    fn progress(&self, _f: f64) { self.progress_calls.fetch_add(1, Ordering::Relaxed); }
}

#[test]
fn disabled_hooks_are_never_invoked() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims = make_prims(512);
    // HIGH would trigger spatial splits IF SPATIAL_SPLITS were true; it is false
    // here.
    let cfg = BuildConfig {
        quality: BuildQuality::HIGH,
        ..Default::default()
    };
    let d = Disabled::default();

    bvh.build_scoped(&cfg, &mut prims, &d, |_r| ()).unwrap();

    assert_eq!(
        d.split_calls.load(Ordering::Relaxed),
        0,
        "split must not run when SPATIAL_SPLITS is false"
    );
    assert_eq!(
        d.progress_calls.load(Ordering::Relaxed),
        0,
        "progress must not run when PROGRESS is false"
    );
}

#[test]
fn result_supports_parallel_traversal() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims = make_prims(256);
    let cfg = BuildConfig::default();
    let recorder = Recorder::default();

    // Inside the scope, traverse each root subtree on its own thread. This only
    // compiles/runs if BvhResult is Sync and Node is Sync (handles are Send).
    let total = bvh
        .build_scoped(&cfg, &mut prims, &recorder, |r| match r.root() {
            None => 0,
            Some(Node::Leaf { prim_count }) => *prim_count,
            Some(Node::Inner { kids, .. }) => std::thread::scope(|s| {
                let handles: Vec<_> = kids
                    .iter()
                    .flatten()
                    .map(|k| {
                        let rr = &r;
                        let kk = *k;
                        s.spawn(move || sum_prims(rr, rr.resolve(kk)))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).sum::<u32>()
            }),
        })
        .unwrap();

    assert_eq!(total, 256, "parallel traversal must visit every primitive");
}
