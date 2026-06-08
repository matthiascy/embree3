# Changelog

All notable changes to this crate are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (while `0.x`, a minor
bump may carry breaking changes).

## [0.4.0] - 2026-06-08

First public release of `embree3` since it was forked from
[Twinklebear/embree-rs](https://github.com/Twinklebear/embree-rs) (originally by
Will Usher). The fork exists to rework the `unsafe` FFI layer of the Embree
3.13.5 bindings into a **memory-safe** Rust API, and this release is the result
of that work. Coming from the upstream crate, treat this as a new API rather
than a drop-in upgrade -- see the notes at the end.

### Safe FFI layer

- Callback closures are heap-owned and reached through a lock-free per-geometry
  callback table, removing the stack-local use-after-free and user-data type
  confusion that the trampoline pattern is prone to.
- A `GeometryBuilder` / `Geometry` typestate enforces Embree's "modify only what
  you uniquely own" rule: all mutation lives on a unique, `!Sync` builder, while
  the committed `Geometry` is read-only and `Send + Sync`.
- `Scene` is a unique, non-`Clone` owner of its handle; mutation and `commit`
  take `&mut self`, so they cannot race a query. Share a committed scene across
  threads with `Arc<Scene>`. These ownership rules are checked at compile time
  (compile-fail test suite).
- A typed, zero-copy buffer API (typed `set_*_buffer` methods and a
  `BufferSource` query) replaces a single untyped buffer enum.

### Features

- **Standalone BVH builder** over `rtcBuildBVH`: `Device::create_bvh` ->
  `Bvh::build_scoped` with a caller-supplied `BvhBuilder` (bring-your-own node
  layout). A generative `for<'id>` brand keeps node handles from escaping the
  build scope.
- **Scene-vs-scene collision**: `Scene::collide` (`rtcCollide`), reporting
  broad-phase candidate primitive pairs via a `Sync` callback. Cheap
  preconditions (same device, non-empty, user-geometry-only) are checked and
  returned as `Err` before reaching Embree.
- **Packet ray queries** with a typed, 16-byte-aligned validity mask
  `ValidMaskN<N>` for `intersect{4,8,16}` / `occluded{4,8,16}` /
  `point_query{4,8,16}`.
- **Ergonomic packet-lane iteration** in user-geometry callbacks
  (`for_each_active_lane` -> `IntersectLane` / `OccludedLane`), with public
  `unsafe` unchecked SoA accessors as a bounds-check escape hatch for filter
  callbacks iterating proven-in-range lanes.
- **Per-ray context extensions** recovered inside callbacks via the `unsafe`
  `IntersectContext::ext` / `IntersectFunctionNArgs::context_ext` accessors,
  keeping the recovery localized to the callback that knows the extension type.

### Testing

A rendered image does not prove memory safety, so soundness is checked directly:
soundness-proof tests register state-capturing closures and assert on the
captured state; compile-fail (trybuild) tests assert the compile-time guarantees
on a pinned toolchain; and CI runs AddressSanitizer + LeakSanitizer (gate) and
ThreadSanitizer (advisory) over the FFI layer that Miri cannot reach (Miri is
scoped to pure-Rust `--lib`).

### Notes for users coming from upstream `embree-rs`

The safe API differs substantially from the original bindings. Notably: `Scene`
is no longer `Clone` (share via `Arc<Scene>`); geometry mutation goes through the
`GeometryBuilder` typestate; user-geometry and filter callbacks no longer carry a
context-type generic (recover a per-ray extension with the `unsafe` `ext` /
`context_ext` accessors); packet queries take `ValidMaskN` rather than
`[i32; N]`; and a few FFI-boundary calls (`Scene::collide`,
`Scene::join_commit`) are `unsafe` with documented contracts. There is no
automated migration path -- see the [README](README.md), the crate docs, and
[`examples/`](examples/) for current usage.
