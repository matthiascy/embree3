# embree

Rust bindings to [Embree](https://embree.github.io/), Intel's high-performance
ray-tracing kernels. Targets **Embree 3.13.5**.

> **This is a fork** of [Twinklebear/embree-rs](https://github.com/Twinklebear/embree-rs)
> (originally by Will Usher), reworking the `unsafe` FFI layer for memory safety:
> heap-owned callback closures reached through a lock-free per-geometry callback
> table, a `GeometryBuilder`/`Geometry` typestate that enforces embree's "modify
> only what you uniquely own" rule, and a typed, zero-copy buffer API.

[![CI](https://github.com/matthiascy/embree-rs/actions/workflows/main.yml/badge.svg)](https://github.com/matthiascy/embree-rs/actions/workflows/main.yml)

Still in development; some features are in progress.

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
