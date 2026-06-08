# embree3

Rust bindings to [Embree](https://embree.github.io/), Intel's high-performance
ray-tracing kernels. Targets **Embree 3.13.5**.

> **This is a fork** of [Twinklebear/embree-rs](https://github.com/Twinklebear/embree-rs)
> (originally by Will Usher), reworking the `unsafe` FFI layer for memory safety:
> heap-owned callback closures reached through a lock-free per-geometry callback
> table, a `GeometryBuilder`/`Geometry` typestate that enforces embree's "modify
> only what you uniquely own" rule, and a typed, zero-copy buffer API.

[![CI](https://github.com/matthiascy/embree3/actions/workflows/main.yml/badge.svg)](https://github.com/matthiascy/embree3/actions/workflows/main.yml)

First public release. The core Embree 3 API is wrapped behind a safe, tested
interface and is usable today. It is pre-1.0, so the API may still change between
minor releases, and some Embree functions are not yet wrapped.

## Features

- **Safe, capturing callbacks**: user-geometry intersect/occluded, filter,
  bounds, displacement, and progress callbacks take ordinary Rust closures,
  bridged to Embree with no leaks or use-after-free.
- **Single, packet, and stream ray queries**: `intersect`/`occluded`, the
  `{4,8,16}`-wide packet variants with a typed, 16-byte-aligned `ValidMaskN`, and
  the stream APIs; per-ray context extensions are recoverable inside callbacks.
- **User geometry** with ergonomic lane iteration (`for_each_active_lane`) and an
  unchecked fast path for hot filter loops.
- **Standalone BVH builder** (`Device::create_bvh`, `Bvh::build_scoped`):
  bring-your-own node layout over `rtcBuildBVH`, with handles confined to the
  build scope.
- **Scene-vs-scene collision** (`Scene::collide`) for broad-phase candidate pairs.
- **Typed, zero-copy buffers** across Embree-managed, geometry-local, and shared
  host memory.

## Building

The build links the native `embree3` library, so it needs `EMBREE_DIR` pointing
at an Embree 3.13.5 install (a directory containing `lib/`):

```bash
wget https://github.com/embree/embree/releases/download/v3.13.5/embree-3.13.5.x86_64.linux.tar.gz
tar -xf embree-3.13.5.x86_64.linux.tar.gz
export EMBREE_DIR="$PWD/embree-3.13.5.x86_64.linux/"
export LD_LIBRARY_PATH="$EMBREE_DIR/lib:$LD_LIBRARY_PATH"

cargo build
cargo test
```

## Documentation

Build the API docs locally with `cargo doc --open`. Embree's own API reference is
[here](https://embree.github.io/api.html), and [`examples/`](examples/) has sample
applications using the bindings.

## License

MIT - see [`LICENSE.md`](LICENSE.md). The native Embree library is licensed
separately (Apache-2.0) by Intel.
