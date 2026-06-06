//! `BuildConfig::validate` + ZST rejection: every bad input must return
//! `Err(Error::INVALID_ARGUMENT)` before any FFI call.
use embree3::{
    Allocator, BuildConfig, BuildPrimitive, BuildQuality, BvhBuilder, BvhNode, ChildBounds,
    Children, Device, Error,
};

// Node types are LOCAL newtypes: `unsafe impl BvhNode for u32`/`()` would
// violate the orphan rule in this integration-test crate (both the trait and
// those types are foreign).

/// A non-ZST scalar node for the config-rejection builders (callbacks never
/// run: validation fails first).
#[derive(Clone, Copy)]
struct Scalar(u32);
unsafe impl BvhNode for Scalar {}

/// A zero-sized node, to test the size-of-node rejection.
#[derive(Clone, Copy)]
struct ZstNode;
unsafe impl BvhNode for ZstNode {}

/// `MAX_CHILDREN = 17` so the branching-cap checks (8 for LOW, 16 for SAH) are
/// not pre-empted by the `> MAX_CHILDREN` check.
struct Wide;
impl BvhBuilder for Wide {
    type Node<'id> = Scalar;
    const MAX_CHILDREN: usize = 17;
    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Scalar {
        a.alloc(Scalar(0))
    }
    fn set_children<'id>(&self, _n: &mut Scalar, _c: Children<'id, Scalar>) {}
    fn set_bounds<'id>(&self, _n: &mut Scalar, _b: ChildBounds<'_>) {}
    fn create_leaf<'id>(&self, a: &Allocator<'id>, _p: &[BuildPrimitive]) -> &'id mut Scalar {
        a.alloc(Scalar(0))
    }
}

/// A builder with a narrow child array, to test the `> MAX_CHILDREN` rejection.
struct Narrow;
impl BvhBuilder for Narrow {
    type Node<'id> = Scalar;
    const MAX_CHILDREN: usize = 2;
    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Scalar {
        a.alloc(Scalar(0))
    }
    fn set_children<'id>(&self, _n: &mut Scalar, _c: Children<'id, Scalar>) {}
    fn set_bounds<'id>(&self, _n: &mut Scalar, _b: ChildBounds<'_>) {}
    fn create_leaf<'id>(&self, a: &Allocator<'id>, _p: &[BuildPrimitive]) -> &'id mut Scalar {
        a.alloc(Scalar(0))
    }
}

/// A builder whose `Node` is a ZST, to test the size-of-node rejection.
struct Zst;
impl BvhBuilder for Zst {
    type Node<'id> = ZstNode;
    const MAX_CHILDREN: usize = 2;
    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut ZstNode {
        a.alloc(ZstNode)
    }
    fn set_children<'id>(&self, _n: &mut ZstNode, _c: Children<'id, ZstNode>) {}
    fn set_bounds<'id>(&self, _n: &mut ZstNode, _b: ChildBounds<'_>) {}
    fn create_leaf<'id>(&self, a: &Allocator<'id>, _p: &[BuildPrimitive]) -> &'id mut ZstNode {
        a.alloc(ZstNode)
    }
}

fn prims(n: u32) -> Vec<BuildPrimitive> {
    (0..n)
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
        .collect()
}

#[test]
fn rejects_invalid_configs_with_wide_builder() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let d = BuildConfig::default();

    macro_rules! rejects {
        ($cfg:expr) => {{
            let mut p = prims(8);
            assert_eq!(
                bvh.build_scoped(&$cfg, &mut p, &Wide, |_r| ()),
                Err(Error::INVALID_ARGUMENT),
            );
        }};
    }

    rejects!(BuildConfig {
        quality: BuildQuality::REFIT,
        ..d.clone()
    });
    rejects!(BuildConfig {
        max_branching_factor: 1,
        ..d.clone()
    }); // < 2
    rejects!(BuildConfig {
        max_branching_factor: 17,
        ..d.clone()
    }); // > 16 (SAH cap)
    rejects!(BuildConfig {
        quality: BuildQuality::LOW,
        max_branching_factor: 9,
        ..d.clone()
    }); // > 8 (Morton cap)
    rejects!(BuildConfig {
        max_leaf_size: 0,
        ..d.clone()
    });
    rejects!(BuildConfig {
        max_leaf_size: 33,
        ..d.clone()
    }); // > 32
    rejects!(BuildConfig {
        min_leaf_size: 5,
        max_leaf_size: 4,
        ..d.clone()
    });
    rejects!(BuildConfig {
        sah_block_size: 0,
        ..d.clone()
    });
    rejects!(BuildConfig {
        max_depth: 0,
        ..d.clone()
    });
    rejects!(BuildConfig {
        traversal_cost: f32::NAN,
        ..d.clone()
    });
    rejects!(BuildConfig {
        traversal_cost: 0.0,
        ..d.clone()
    });
    rejects!(BuildConfig {
        intersection_cost: -1.0,
        ..d.clone()
    });
    // No flags case: the single build flag is `dynamic: bool`, which cannot be
    // invalid.
}

#[test]
fn rejects_branching_above_max_children() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut p = prims(8);
    // Narrow::MAX_CHILDREN == 2, so branching 3 is rejected even though 3 <= 16.
    let cfg = BuildConfig {
        max_branching_factor: 3,
        ..Default::default()
    };
    assert_eq!(
        bvh.build_scoped(&cfg, &mut p, &Narrow, |_r| ()),
        Err(Error::INVALID_ARGUMENT),
    );
}

#[test]
fn rejects_zero_sized_node_type() {
    let device = Device::new().unwrap();
    let mut bvh = device.create_bvh().unwrap();
    let mut p = prims(8);
    let cfg = BuildConfig::default();
    assert_eq!(
        bvh.build_scoped(&cfg, &mut p, &Zst, |_r| ()),
        Err(Error::INVALID_ARGUMENT),
    );
}
