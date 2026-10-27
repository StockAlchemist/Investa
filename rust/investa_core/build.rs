//! Embeds a hash of the crate's sources as `INVESTA_CORE_SOURCE_HASH`.
//!
//! The Python side folds it into `portfolio_version.CURRENT_HIST_VERSION`, so the
//! on-disk history caches invalidate when a kernel changes, exactly as they do
//! when a Python engine file changes. It has to be baked in at build time: the
//! runtime image ships the compiled module, not these sources.

use std::fs;
use std::path::Path;

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= u64::from(*b);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

fn main() {
    let mut files: Vec<_> = fs::read_dir("src")
        .expect("src/ is readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        fnv1a(&mut hash, name.as_bytes());
        fnv1a(&mut hash, &fs::read(path).expect("source is readable"));
    }
    println!("cargo:rustc-env=INVESTA_CORE_SOURCE_HASH={hash:016x}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed={}", Path::new("build.rs").display());
}
