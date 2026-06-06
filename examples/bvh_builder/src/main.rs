//! Builds a BVH over a handful of triangle AABBs with embree's standalone
//! builder, then does a recursive ray-vs-AABB descent collecting candidate
//! primitives. Mirrors embree's `bvh_builder` tutorial.
//!
//! Run with EMBREE_DIR + LD_LIBRARY_PATH set:
//!   cargo run -p bvh_builder
use embree3::{
    Allocator, Bounds, BuildConfig, BuildPrimitive, BvhBuilder, BvhNode, BvhResult, ChildBounds,
    Children, Device, NodePtr,
};

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

#[derive(Clone, Copy)]
enum Node<'id> {
    Inner {
        bounds: [Bounds; 2],
        kids: [Option<NodePtr<'id, Node<'id>>>; 2],
    },
    Leaf {
        prim_id: u32,
    },
}
unsafe impl<'id> BvhNode for Node<'id> {}

struct Builder;
impl BvhBuilder for Builder {
    type Node<'id> = Node<'id>;
    const MAX_CHILDREN: usize = 2;

    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Node<'id> {
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
        // This example builds with `max_leaf_size = 1`, so each leaf holds exactly one
        // primitive and `prims[0]` loses nothing. (With a larger leaf size, a leaf
        // would need to represent every primitive in `prims`, e.g. an inline
        // array.)
        debug_assert_eq!(prims.len(), 1);
        a.alloc(Node::Leaf {
            prim_id: prims[0].primID,
        })
    }
}

/// Slab-style ray-vs-AABB test; returns true if the ray [0, t_max) hits the
/// box.
fn ray_hits_box(origin: [f32; 3], inv_dir: [f32; 3], b: &Bounds, t_max: f32) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = t_max;
    let lo = [b.lower_x, b.lower_y, b.lower_z];
    let hi = [b.upper_x, b.upper_y, b.upper_z];
    for axis in 0..3 {
        let t1 = (lo[axis] - origin[axis]) * inv_dir[axis];
        let t2 = (hi[axis] - origin[axis]) * inv_dir[axis];
        tmin = tmin.max(t1.min(t2));
        tmax = tmax.min(t1.max(t2));
    }
    tmin <= tmax
}

fn collect_candidates<'id>(
    r: &BvhResult<'id, Builder>,
    node: &Node<'id>,
    origin: [f32; 3],
    inv_dir: [f32; 3],
    out: &mut Vec<u32>,
) {
    match node {
        Node::Leaf { prim_id } => out.push(*prim_id),
        Node::Inner { bounds, kids } => {
            for (slot, child) in kids.iter().enumerate() {
                if let Some(k) = child {
                    if ray_hits_box(origin, inv_dir, &bounds[slot], f32::INFINITY) {
                        collect_candidates(r, r.resolve(*k), origin, inv_dir, out);
                    }
                }
            }
        }
    }
}

fn main() {
    let device = Device::new().expect("create device");
    let mut bvh = device.create_bvh().expect("create bvh");

    // Eight unit boxes along x.
    let mut prims: Vec<BuildPrimitive> = (0..8)
        .map(|i| BuildPrimitive {
            lower_x: i as f32,
            lower_y: 0.0,
            lower_z: 0.0,
            geomID: 0,
            upper_x: i as f32 + 1.0,
            upper_y: 1.0,
            upper_z: 1.0,
            primID: i,
        })
        .collect();

    // A ray along +x at height 0.5 should pass through every box.
    let origin = [-1.0f32, 0.5, 0.5];
    let dir = [1.0f32, 0.0, 0.0];
    let inv_dir = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];

    // One primitive per leaf, so each `Node::Leaf` represents exactly one box.
    let cfg = BuildConfig {
        max_leaf_size: 1,
        ..Default::default()
    };
    let candidates = bvh
        .build_scoped(&cfg, &mut prims, &Builder, |r| {
            let mut out = Vec::new();
            if let Some(root) = r.root() {
                collect_candidates(&r, root, origin, inv_dir, &mut out);
            }
            out.sort_unstable();
            out
        })
        .expect("build");

    println!("ray hit candidate primIDs: {candidates:?}");
    assert_eq!(candidates, (0..8).collect::<Vec<u32>>());
    println!("ok: the +x ray traverses all eight boxes");
}
