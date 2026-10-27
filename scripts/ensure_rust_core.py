#!/usr/bin/env python3
"""Build the Rust calculation core if it is missing or stale.

The backend imports `investa_core`, compiled from rust/investa_core. A build
from older sources would quietly run the old kernels, so this compares the hash
baked into the installed module (see rust/investa_core/build.rs) with the hash
of the sources on disk, and rebuilds on any difference.

Usage: python3 scripts/ensure_rust_core.py [--check]
  --check   report only; exit 1 if a rebuild is needed
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE = os.path.join(ROOT, "rust", "investa_core")


def source_hash() -> str:
    """FNV-1a over the sorted src/*.rs names and contents, exactly as build.rs."""
    h = 0xCBF29CE484222325
    src = os.path.join(CRATE, "src")
    for name in sorted(f for f in os.listdir(src) if f.endswith(".rs")):
        with open(os.path.join(src, name), "rb") as f:
            data = name.encode() + f.read()
        for b in data:
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def installed_hash():
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import investa_core; print(investa_core.SOURCE_HASH)"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def main() -> int:
    want, have = source_hash(), installed_hash()
    if want == have:
        print(f"investa_core is current ({want[:8]})")
        return 0
    print(f"investa_core is {'stale' if have else 'not installed'}: built {have} / sources {want}")
    if "--check" in sys.argv:
        return 1
    if shutil.which("cargo") is None:
        print("cargo not found: install Rust from https://rustup.rs, then re-run", file=sys.stderr)
        return 1
    print("Building investa_core (release)...")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", CRATE]
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0 or installed_hash() != want:
        print("investa_core build failed", file=sys.stderr)
        return 1
    print(f"investa_core built ({want[:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
