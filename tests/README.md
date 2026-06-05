# embree-rs FFI soundness tests

These tests prove the callback/user-data lifetime and type model is sound. A rendered
image is NOT proof; UB does not reliably crash. Each proof captures non-trivial state,
clobbers the stack, and asserts on the captured state.

Requires `EMBREE_DIR` + `LD_LIBRARY_PATH`.
