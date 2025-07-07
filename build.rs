#![allow(unused)]

use std::path::PathBuf;

/// Generate link search paths from a list of paths.
///
/// This allows paths like `/path/to/lib1:/path/to/lib2` to be split into
/// individual paths.
fn generate_link_search_paths(paths: &str) -> Vec<String> {
    let split_char = if cfg!(windows) { ";" } else { ":" };
    paths.split(split_char).map(|path| path.to_string()).collect()
}

/// Generate root candidates for library search paths.
///
/// Code modified from
///
/// https://github.com/coreylowman/cudarc/blob/main/build.rs
fn root_candidates(env_candidates: &[&str]) -> Vec<PathBuf> {
    let root_candidates = ["/usr", "/usr/local", "/usr/local/share", "/opt"];

    env_candidates
        .iter()
        .map(|p| p.to_string())
        .map(std::env::var)
        .filter_map(Result::ok)
        .flat_map(|path| generate_link_search_paths(&path))
        .filter(|path| !path.is_empty())
        .chain(root_candidates.into_iter().map(|p| p.to_string()))
        .map(|p| p.into())
        .collect()
}

/// Generate candidates for library search paths.
///
/// Code modified from
///
/// https://github.com/coreylowman/cudarc/blob/main/build.rs
fn lib_candidates() -> impl Iterator<Item = PathBuf> {
    let lib_candidates = [
        "",
        "lib",
        "lib/stubs",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ];
    lib_candidates.into_iter().map(|p| p.into())
}

fn path_candidates(env_candidates: &[&str]) -> impl Iterator<Item = PathBuf> {
    root_candidates(env_candidates)
        .into_iter()
        .flat_map(|root| lib_candidates().map(move |lib| root.join(lib)))
        .filter(|path| path.exists())
        .map(|path| std::fs::canonicalize(path).unwrap())
}

fn link_openblas() {
    let env_candidates = ["REST_EXT_DIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"];
    // minimal rerun-if-env-changed to avoid unnecessary rebuilds
    for path in path_candidates(&env_candidates) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=gomp");
}

fn main() {
    #[cfg(feature = "use_openblas")]
    link_openblas();
}
