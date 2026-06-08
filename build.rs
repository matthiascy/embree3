use std::{env, path::PathBuf};

fn main() {
    // docs.rs renders the crate's documentation but has no Embree install, and
    // `cargo doc` does not link a library -- so on docs.rs, skip the native link
    // setup entirely. Emitting `cargo:rustc-link-lib=embree3` there would leave
    // an unsatisfiable link requirement. docs.rs sets `DOCS_RS` in the build env.
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    if let Ok(e) = env::var("EMBREE_DIR") {
        let mut embree_dir = PathBuf::from(e);
        embree_dir.push("lib");
        println!("cargo:rustc-link-search=native={}", embree_dir.display());
        println!("cargo:rerun-if-env-changed=EMBREE_DIR");
    }
    println!("cargo:rustc-link-lib=embree3");
}
