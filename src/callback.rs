//! Type-erased, heap-owned storage for FFI callback closures.
//!
//! A closure handed to embree must live at a stable address for as long as
//! embree may call it. `ErasedFn` boxes the closure on the heap and remembers
//! how to drop it without knowing its concrete type at the drop point. The
//! trampoline function is a thin shim that casts the user pointer back to the
//! erased closure and invokes it.

use std::{fmt, os::raw::c_void};

pub(crate) struct ErasedFn {
    ptr: *mut c_void,
    drop_fn: unsafe fn(*mut c_void),
}

impl ErasedFn {
    /// Boxes a closure and returns an `ErasedFn` that can be safely passed to
    /// embree.
    ///
    /// `F` must be `Send + Sync`: the closure is reached from embree worker
    /// threads via a shared `&F` and may be dropped on any thread.
    pub(crate) fn new<F: Send + Sync + 'static>(f: F) -> Self {
        unsafe fn drop_glue<F>(p: *mut c_void) { drop(Box::from_raw(p as *mut F)); }

        Self {
            ptr: Box::into_raw(Box::new(f)) as *mut c_void,
            drop_fn: drop_glue::<F>,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut c_void { self.ptr }
}

// SAFETY: the only constructor requires `F: Send + Sync`; `ptr` uniquely owns
// that boxed `F`, and `drop_fn` is a plain fn pointer. So an `ErasedFn` can be
// sent to / shared with other threads exactly as the boxed `F` can.
unsafe impl Send for ErasedFn {}
unsafe impl Sync for ErasedFn {}

impl Drop for ErasedFn {
    fn drop(&mut self) { unsafe { (self.drop_fn)(self.ptr) } }
}

impl fmt::Debug for ErasedFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErasedFn").field("ptr", &self.ptr).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    /// Round-trips a closure through `ErasedFn` exactly as a trampoline does:
    /// box it, recover the *concrete* `F` from the type-erased pointer as a
    /// shared `&F`, call it, then drop the owner. Recovering as `*const F`
    /// (the real closure type, shared) is the sound cast; a shared `&F` is
    /// what lets concurrent callbacks not alias.
    fn roundtrip<F>(f: F, x: u32) -> u32
    where
        F: Fn(u32) -> u32 + Send + Sync + 'static,
    {
        let erased = ErasedFn::new(f);
        // SAFETY: `as_ptr()` is the `*mut F` that `ErasedFn::new` boxed; still live.
        let f: &F = unsafe { &*(erased.as_ptr() as *const F) };
        f(x)
        // `erased` drops here, freeing the boxed `F` and its captures exactly
        // once.
    }

    #[test]
    fn boxes_calls_and_drops_without_leak() {
        // `witness` observes that the captured state is dropped exactly once: its
        // strong count goes 2 -> 1 when `ErasedFn` frees the closure.
        let witness = Arc::new(AtomicU32::new(0));
        let captured = witness.clone();
        assert_eq!(Arc::strong_count(&witness), 2);

        let out = roundtrip(
            move |x| {
                captured.fetch_add(1, Ordering::Relaxed); // exercise the capture (Fn-compatible)
                x + 1
            },
            41,
        );

        assert_eq!(out, 42, "recovered closure must run with correct behaviour");
        assert_eq!(
            witness.load(Ordering::Relaxed),
            1,
            "closure body must have executed once"
        );
        assert_eq!(
            Arc::strong_count(&witness),
            1,
            "ErasedFn::drop must release the closure's captures exactly once"
        );
    }
}
