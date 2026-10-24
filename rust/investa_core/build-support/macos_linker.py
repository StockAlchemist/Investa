#!/usr/bin/env python3
"""Linker wrapper for macOS: link with `cc`, then fix the string-pool alignment.

Why: the Xcode 27 linker (ld-27037) writes LINKEDIT's string pool straight
after the indirect symbol table, with no padding. That table holds 4-byte
entries, so an odd count leaves the pool 4 bytes off an 8-byte boundary, and
the macOS 27 loader refuses the result: "mis-aligned LINKEDIT string pool".
Whether a build hits it depends on how many symbols it imports, so it comes
and goes as code changes. `-ld_classic` is gone, and rust-lld cannot read the
macOS 27 SDK's .tbd stubs, so there is no linker to switch to.

The fix, applied only when the pool is misaligned:
  1. drop the linker's ad-hoc signature (it covers the bytes about to move),
  2. insert the padding the linker left out, just before the string pool,
  3. shift every LINKEDIT offset at or past that point, grow __LINKEDIT,
  4. ad-hoc sign again, as arm64 requires.

Cargo runs this for macOS targets (rust/investa_core/.cargo/config.toml); it
is a no-op for a correctly aligned binary. Usable by hand too:
    build-support/macos_linker.py --fix path/to/lib.dylib
"""

import os
import struct
import subprocess
import sys

MH_MAGIC_64 = 0xFEEDFACF
LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x2
LC_DYSYMTAB = 0xB
LC_CODE_SIGNATURE = 0x1D
LC_DYLD_INFO = 0x22
LC_DYLD_INFO_ONLY = 0x80000022
# linkedit_data_command: cmd, cmdsize, dataoff, datasize
LINKEDIT_DATA = {
    0x1D,  # LC_CODE_SIGNATURE
    0x1E,  # LC_SEGMENT_SPLIT_INFO
    0x26,  # LC_FUNCTION_STARTS
    0x29,  # LC_DATA_IN_CODE
    0x2B,  # LC_DYLIB_CODE_SIGN_DRS
    0x2E,  # LC_LINKER_OPTIMIZATION_HINT
    0x80000033,  # LC_DYLD_EXPORTS_TRIE
    0x80000034,  # LC_DYLD_CHAINED_FIXUPS
}
ALIGN = 8
PAGE = 0x4000


def _load_commands(data):
    magic, _cpu, _sub, _ftype, ncmds, _size, _flags, _res = struct.unpack_from("<8I", data, 0)
    if magic != MH_MAGIC_64:
        return None
    off = 32
    cmds = []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<2I", data, off)
        cmds.append((cmd, off, cmdsize))
        off += cmdsize
    return cmds


def _stroff(data):
    for cmd, off, _ in _load_commands(data) or []:
        if cmd == LC_SYMTAB:
            return struct.unpack_from("<I", data, off + 16)[0]
    return None


def fix(path):
    """Realign the string pool of the Mach-O at `path`. Returns True if changed."""
    with open(path, "rb") as f:
        data = f.read()
    cmds = _load_commands(data)
    stroff = _stroff(data) if cmds is not None else None
    if stroff is None or stroff % ALIGN == 0:
        return False

    if any(cmd == LC_CODE_SIGNATURE for cmd, _, _ in cmds):
        subprocess.run(["codesign", "--remove-signature", path], check=True)
        with open(path, "rb") as f:
            data = f.read()
        cmds = _load_commands(data)
        stroff = _stroff(data)

    pad = (-stroff) % ALIGN
    buf = bytearray(data[:stroff] + b"\0" * pad + data[stroff:])

    def shift(field_off):
        value = struct.unpack_from("<I", buf, field_off)[0]
        if value and value >= stroff:
            struct.pack_into("<I", buf, field_off, value + pad)

    for cmd, off, _ in cmds:
        if cmd == LC_SYMTAB:
            # symoff precedes the pool; stroff is the pool itself.
            struct.pack_into("<I", buf, off + 16, stroff + pad)
        elif cmd == LC_DYSYMTAB:
            # tocoff, modtaboff, extrefsymoff, indirectsymoff, extreloff, locreloff
            for field in (32, 40, 48, 56, 64, 72):
                shift(off + field)
        elif cmd in (LC_DYLD_INFO, LC_DYLD_INFO_ONLY):
            for field in (8, 16, 24, 32, 40):  # rebase, bind, weak, lazy, export
                shift(off + field)
        elif cmd in LINKEDIT_DATA:
            shift(off + 8)
        elif cmd == LC_SEGMENT_64:
            segname = bytes(buf[off + 8 : off + 24]).rstrip(b"\0")
            if segname == b"__LINKEDIT":
                vmsize, fileoff, filesize = struct.unpack_from("<3Q", buf, off + 32)
                filesize += pad
                vmsize = max(vmsize, (filesize + PAGE - 1) // PAGE * PAGE)
                struct.pack_into("<3Q", buf, off + 32, vmsize, fileoff, filesize)

    with open(path, "wb") as f:
        f.write(buf)
    subprocess.run(["codesign", "--force", "--sign", "-", path], check=True)
    with open(path, "rb") as f:
        if _stroff(f.read()) % ALIGN:
            raise SystemExit(f"{path}: string pool still misaligned after fix")
    return True


def _output_path(args):
    """The `-o` argument, looking inside @response files rustc may pass."""
    expanded = []
    for a in args:
        if a.startswith("@") and os.path.exists(a[1:]):
            with open(a[1:]) as f:
                expanded.extend(line.rstrip("\n") for line in f)
        else:
            expanded.append(a)
    for i, a in enumerate(expanded):
        if a == "-o" and i + 1 < len(expanded):
            return expanded[i + 1]
    return None


def main(argv):
    if argv[:1] == ["--fix"]:
        for path in argv[1:]:
            print(f"{path}: {'realigned' if fix(path) else 'already aligned'}")
        return 0
    rc = subprocess.call([os.environ.get("INVESTA_CC", "cc"), *argv])
    if rc != 0:
        return rc
    out = _output_path(argv)
    if out and os.path.exists(out) and out.endswith((".dylib", ".so")):
        fix(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
