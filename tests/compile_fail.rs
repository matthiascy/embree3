//! Compile-fail proofs for the crate's *compile-time* soundness guarantees.
//!
//! These matter as much as the runtime soundness tests: the whole point of the
//! API design is that certain unsound patterns must not even compile. A test
//! that renders an image proves nothing about a use-after-free; a test that
//! *fails to compile* proves the type system rules the bug out entirely. Each
//! case under `tests/compile_fail/` is expected to FAIL to compile, with the
//! exact error captured in its sibling `.stderr` file.
//!
//! # Regenerating the `.stderr` snapshots
//!
//! The `.stderr` files are produced by the compiler, not hand-written. After a
//! deliberate API change (or on first run), regenerate them with:
//!
//! ```text
//! TRYBUILD=overwrite cargo test --test compile_fail
//! ```

#[test]
fn compile_fail_guards() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
