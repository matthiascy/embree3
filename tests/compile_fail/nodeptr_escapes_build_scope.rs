//! A `NodePtr` handle cannot escape the `build_scoped` generative scope.
//! `build_scoped` takes a `for<'id> FnOnce(BvhResult<'id, B>) -> R` closure, so
//! the brand `'id` is fresh per call and the return type `R` cannot mention it.
//! Returning a handle would let it be resolved against a *different* build's
//! arena (or one already freed), so the compiler must reject it.
#![allow(dead_code)]

use embree3::{
    Allocator, BuildConfig, BuildPrimitive, BvhBuilder, BvhNode, ChildBounds, Children, Device,
    NodePtr,
};

#[derive(Clone, Copy)]
enum Node<'id> {
    Inner {
        kids: [Option<NodePtr<'id, Node<'id>>>; 2],
    },
    Leaf {
        prim_id: u32,
    },
}
unsafe impl<'id> BvhNode for Node<'id> {}

struct B;
impl BvhBuilder for B {
    type Node<'id> = Node<'id>;
    const MAX_CHILDREN: usize = 2;

    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Node<'id> {
        a.alloc(Node::Inner { kids: [None; 2] })
    }
    fn set_children<'id>(&self, node: &mut Node<'id>, c: Children<'id, Node<'id>>) {
        if let Node::Inner { kids } = node {
            for i in 0..c.len().min(2) {
                kids[i] = c.get(i);
            }
        }
    }
    fn set_bounds<'id>(&self, _node: &mut Node<'id>, _b: ChildBounds<'_>) {}
    fn create_leaf<'id>(&self, a: &Allocator<'id>, prims: &[BuildPrimitive]) -> &'id mut Node<'id> {
        a.alloc(Node::Leaf {
            prim_id: prims[0].primID,
        })
    }
}

fn main() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut prims: Vec<BuildPrimitive> = Vec::new();
    let cfg = BuildConfig {
        max_leaf_size: 1,
        ..Default::default()
    };
    // ERROR: the closure tries to return a `NodePtr<'id, ..>` out of the
    // `for<'id>` scope, so no single `R` satisfies the bound.
    let _escaped = bvh
        .build_scoped(&cfg, &mut prims, &B, |r| r.root_ptr())
        .unwrap();
}
