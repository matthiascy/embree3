//! Traversal cost over a branded 8-byte-handle node tree vs a 16-byte (padded)
//! node tree, to quantify the cache-density argument for the handle width.
//! Both trees are built from the same primitives; the bench measures a full
//! recursive prim-count traversal.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
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

// 8-byte handles
#[derive(Clone, Copy)]
enum Slim<'id> {
    Inner {
        bounds: [Bounds; 2],
        kids: [Option<NodePtr<'id, Slim<'id>>>; 2],
    },
    Leaf {
        prim_count: u32,
    },
}
unsafe impl<'id> BvhNode for Slim<'id> {}

struct SlimB;
impl BvhBuilder for SlimB {
    type Node<'id> = Slim<'id>;
    const MAX_CHILDREN: usize = 2;
    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut Slim<'id> {
        a.alloc(Slim::Inner {
            bounds: [EMPTY_BOUNDS; 2],
            kids: [None; 2],
        })
    }
    fn set_children<'id>(&self, node: &mut Slim<'id>, c: Children<'id, Slim<'id>>) {
        if let Slim::Inner { kids, .. } = node {
            for i in 0..c.len().min(2) {
                kids[i] = c.get(i);
            }
        }
    }
    fn set_bounds<'id>(&self, node: &mut Slim<'id>, bnds: ChildBounds<'_>) {
        if let Slim::Inner { bounds, .. } = node {
            for i in 0..bnds.len().min(2) {
                if let Some(cb) = bnds.get(i) {
                    bounds[i] = *cb;
                }
            }
        }
    }
    fn create_leaf<'id>(&self, a: &Allocator<'id>, prims: &[BuildPrimitive]) -> &'id mut Slim<'id> {
        a.alloc(Slim::Leaf {
            prim_count: prims.len() as u32,
        })
    }
}

fn walk_slim<'id>(r: &BvhResult<'id, SlimB>, n: &Slim<'id>) -> u32 {
    match n {
        Slim::Leaf { prim_count } => *prim_count,
        Slim::Inner { kids, .. } => kids
            .iter()
            .flatten()
            .map(|k| walk_slim(r, r.resolve(*k)))
            .sum(),
    }
}

// 16-byte handles (each child carries an extra word, simulating runtime ids)
#[derive(Clone, Copy)]
struct Wide<'id> {
    h: Option<NodePtr<'id, WideNode<'id>>>,
    _pad: u64,
}
#[derive(Clone, Copy)]
enum WideNode<'id> {
    Inner {
        bounds: [Bounds; 2],
        kids: [Wide<'id>; 2],
    },
    Leaf {
        prim_count: u32,
    },
}
unsafe impl<'id> BvhNode for WideNode<'id> {}

struct WideB;
impl BvhBuilder for WideB {
    type Node<'id> = WideNode<'id>;
    const MAX_CHILDREN: usize = 2;
    fn create_node<'id>(&self, a: &Allocator<'id>, _n: usize) -> &'id mut WideNode<'id> {
        a.alloc(WideNode::Inner {
            bounds: [EMPTY_BOUNDS; 2],
            kids: [Wide { h: None, _pad: 0 }; 2],
        })
    }
    fn set_children<'id>(&self, node: &mut WideNode<'id>, c: Children<'id, WideNode<'id>>) {
        if let WideNode::Inner { kids, .. } = node {
            for i in 0..c.len().min(2) {
                kids[i] = Wide {
                    h: c.get(i),
                    _pad: 0,
                };
            }
        }
    }
    fn set_bounds<'id>(&self, node: &mut WideNode<'id>, bnds: ChildBounds<'_>) {
        if let WideNode::Inner { bounds, .. } = node {
            for i in 0..bnds.len().min(2) {
                if let Some(cb) = bnds.get(i) {
                    bounds[i] = *cb;
                }
            }
        }
    }
    fn create_leaf<'id>(
        &self,
        a: &Allocator<'id>,
        prims: &[BuildPrimitive],
    ) -> &'id mut WideNode<'id> {
        a.alloc(WideNode::Leaf {
            prim_count: prims.len() as u32,
        })
    }
}

fn walk_wide<'id>(r: &BvhResult<'id, WideB>, n: &WideNode<'id>) -> u32 {
    match n {
        WideNode::Leaf { prim_count } => *prim_count,
        WideNode::Inner { kids, .. } => kids
            .iter()
            .filter_map(|w| w.h)
            .map(|k| walk_wide(r, r.resolve(k)))
            .sum(),
    }
}

fn make_prims(n: u32) -> Vec<BuildPrimitive> {
    (0..n)
        .map(|i| BuildPrimitive {
            lower_x: (i % 256) as f32,
            lower_y: (i / 256) as f32,
            lower_z: 0.0,
            geomID: 0,
            upper_x: (i % 256) as f32 + 1.0,
            upper_y: (i / 256) as f32 + 1.0,
            upper_z: 1.0,
            primID: i,
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    // Lock in the size difference this benchmark exists to measure: an 8-byte
    // branded handle vs a 16-byte padded child. (Lifetimes do not affect size;
    // `'static` is an arbitrary concrete choice for naming the types.)
    assert_eq!(
        std::mem::size_of::<Option<NodePtr<'static, Slim<'static>>>>(),
        8,
        "branded handle must stay 8 bytes"
    );
    assert_eq!(
        std::mem::size_of::<Wide<'static>>(),
        16,
        "padded child must be 16 bytes"
    );

    let device = Device::new().expect("device");
    let cfg = BuildConfig::default();
    const N: u32 = 100_000;

    let mut g = c.benchmark_group("bvh_traversal");
    // Report per-primitive throughput (elements/sec) so slim vs wide are directly
    // comparable. NOTE: this measures wall-time/throughput only; cycles-per-node
    // and cache-miss counts require an external profiler (e.g. `perf stat`,
    // `cachegrind`), which is out of scope for this criterion harness.
    g.throughput(Throughput::Elements(N as u64));

    g.bench_function("slim_8byte_handles", |b| {
        let mut bvh = device.create_bvh().unwrap();
        let mut prims = make_prims(N);
        bvh.build_scoped(&cfg, &mut prims, &SlimB, |r| {
            let root = r.root().unwrap();
            b.iter(|| black_box(walk_slim(&r, root)));
        })
        .unwrap();
    });

    g.bench_function("wide_16byte_handles", |b| {
        let mut bvh = device.create_bvh().unwrap();
        let mut prims = make_prims(N);
        bvh.build_scoped(&cfg, &mut prims, &WideB, |r| {
            let root = r.root().unwrap();
            b.iter(|| black_box(walk_wide(&r, root)));
        })
        .unwrap();
    });

    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
