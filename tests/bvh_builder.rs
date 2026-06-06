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
    Allocator, Bounds, BuildConfig, BuildPrimitive, BvhBuilder, BvhNode, BvhResult, ChildBounds,
    Children, Device, NodePtr,
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
