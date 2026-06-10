#!/usr/bin/env python3
"""Patch ELF e_flags in a CUDA cubin from sm_100a (0x6006402) to sm_120a
(0x6007802). Driver checks e_flags to match device arch and accepts/rejects
on that basis. After patching, driver will accept the cubin as sm_120
compatible, attempt to load, and hand SASS bytes to the hardware decoder.
The decoder on sm_120 hardware will then either execute (impossible if
UTCATOMSWS isn't in its ISA) or fault with CUDA_ERROR_ILLEGAL_INSTRUCTION
(715) — which is the empirical hardware test.
"""

import sys, struct

def main():
    if len(sys.argv) < 3:
        print("usage: patch_cubin_arch.py <in.cubin> <out.cubin>")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        data = bytearray(f.read())
    # ELF64 e_flags at offset 0x30, 4 bytes, little-endian.
    EF_OFF = 0x30
    old, = struct.unpack_from("<I", data, EF_OFF)
    print(f"  old e_flags = 0x{old:08x}")
    # Patch:
    #   bits 8..15 of e_flags = SM number (100 → 0x64, 120 → 0x78)
    new = (old & ~0xFF00) | (120 << 8)
    print(f"  new e_flags = 0x{new:08x}")
    struct.pack_into("<I", data, EF_OFF, new)
    # Also need to patch any .nv.info sections that may reference sm_100.
    # cuobjdump showed:
    #   .headerflags @"EF_CUDA_ACCELERATORS EF_CUDA_SM100 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM100)"
    # The virtual SM is encoded somewhere else, let's keep it for now and see
    # if driver complains about a mismatch.
    with open(sys.argv[2], "wb") as f:
        f.write(data)
    print(f"  wrote {sys.argv[2]} ({len(data)} bytes)")

if __name__ == "__main__":
    main()
