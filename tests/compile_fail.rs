//! Compile-fail proofs for the crate's *compile-time* soundness guarantees.
//!
//! These matter as much as the runtime soundness tests: the whole point of the
//! API design is that certain unsound patterns must not even compile. A test
//! that renders an image proves nothing about a use-after-free; a test that
//! *fails to compile* proves the type system rules the bug out entirely. Each
//! case under `tests/compile_fail/` is expected to FAIL to compile, with the
//! exact error captured in its sibling `.stderr` file.
//!
//! # Opt-in: pinned toolchain only
//!
//! trybuild compares *normalized compiler output* exactly, and that wording
//! drifts between rustc versions and channels. The default `cargo test` gate
//! (CI's `build_linux`/`build_mac`/`build_windows`) runs on a floating
//! `stable`, so running this there would spuriously fail whenever the snapshots
//! were generated on a different toolchain. It is therefore **gated behind the
//! `RUN_UI_TESTS` env var** and exercised only by the dedicated, version-pinned
//! `compile_fail` CI job (see `.github/workflows/main.yml`). Without the var
//! the test is a no-op, so it never gates an ordinary build.
//!
//! # Regenerating the `.stderr` snapshots
//!
//! The `.stderr` files are produced by the compiler, not hand-written. On first
//! run, after a deliberate API change, or after bumping the pinned toolchain,
//! regenerate them **on that same pinned toolchain** (needs `EMBREE_DIR`):
//!
//! ```text
//! RUN_UI_TESTS=1 TRYBUILD=overwrite cargo test --test compile_fail
//! ```
//!
//! then review the diff before committing.

#[test]
fn compile_fail_guards() {
    // Opt-in only -- exact-match compiler output is toolchain-specific; see the
    // module docs. Keeps the default `cargo test` (floating `stable`) green.
    if std::env::var_os("RUN_UI_TESTS").is_none() {
        eprintln!(
            "skipping compile-fail tests; set RUN_UI_TESTS=1 to run them (pinned toolchain only)"
        );
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
