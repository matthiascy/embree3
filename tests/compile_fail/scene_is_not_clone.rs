//! `Scene` is deliberately NOT `Clone`. The old `rtcRetainScene`-based `Clone`
//! let several handles alias one native scene, which made `&mut Scene`
//! non-exclusive and opened a use-after-free between a clone replacing the
//! progress callback and another clone's `commit` invoking it. Sharing is via
//! `Arc<Scene>` instead. This must not compile.
//!
//! Asserted through a trait-bound helper rather than `scene.clone()`: a bare
//! `.clone()` on a non-`Clone` value silently autorefs and clones the *`&Scene`
//! reference* instead of erroring, so it would NOT prove the guarantee.
use embree3::Scene;

fn assert_clone<T: Clone>() {}

fn main() { assert_clone::<Scene<'static>>(); }
