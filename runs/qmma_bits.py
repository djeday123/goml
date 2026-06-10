#!/usr/bin/env python3
"""
Analyze QMMA opcode bit layout from sm120_cache.txt.

Goal: find which bit field encodes SHAPE (16816 vs 16832 vs INVALID*)
and which encodes ACC type (F16 vs F32), input types (E4M3/E5M2/E2M3),
SF (scaled), SP (sparse). Then determine whether INVALID enums are
sparse 1-bit deviations from named values (= likely undefined behavior
in a real silicon field) or wholly separate bit codes.
"""
import re
import sys
from collections import defaultdict

CACHE = "/data/lib/podman-data/projects/goml/runs/sm120_cache.txt"

PRED_RE = re.compile(r"^@!?U?P[T0-9]*\s+")

# Collect all QMMA entries: (asm_full, mnemonic, hex16)
entries = []
with open(CACHE) as f:
    for line in f:
        if "---" not in line:
            continue
        asm, hexs = line.split("---", 1)
        asm = PRED_RE.sub("", asm.strip()).strip().rstrip(";")
        if not asm.startswith("QMMA"):
            continue
        tok = asm.split()[0]
        hexs = hexs.strip()
        if len(hexs) != 32:
            continue
        entries.append((asm, tok, hexs))

# Group by mnemonic, keep the FIRST (canonical) hex for each
canon = {}
for asm, mn, hx in entries:
    if mn not in canon:
        canon[mn] = (asm, hx)

# Reference: the most common variant
REF = "QMMA.16816.F16.E4M3.E4M3"
if REF not in canon:
    print(f"reference {REF} not found")
    sys.exit(1)

ref_asm, ref_hex = canon[REF]
ref_bytes = bytes.fromhex(ref_hex)

def bit_diff(a_hex, b_hex):
    a = bytes.fromhex(a_hex)
    b = bytes.fromhex(b_hex)
    diff = []
    for byte_i in range(16):
        x = a[byte_i] ^ b[byte_i]
        for bit_i in range(8):
            if (x >> bit_i) & 1:
                # global bit index = byte_i*8 + bit_i
                diff.append(byte_i * 8 + bit_i)
    return diff

def bit_value(hx, bit_idx):
    b = bytes.fromhex(hx)
    return (b[bit_idx // 8] >> (bit_idx % 8)) & 1

print(f"=== Reference: {REF} ===")
print(f"hex = {ref_hex}")
print(f"bin (little-endian byte order, LSB-first per byte):")
ref_bits = ''.join(f"{ref_bytes[i]:08b}"[::-1] for i in range(16))
print(f"  {ref_bits[:32]}")
print(f"  {ref_bits[32:64]}")
print(f"  {ref_bits[64:96]}")
print(f"  {ref_bits[96:]}")
print()

# Print bit-diff vs reference for every other QMMA variant.
# Operands are all R0,R0,R0,R0 so non-opcode bits should be zero
# in the reference; diffs locate the opcode field encoding.
print(f"=== Bit-diff each QMMA variant vs {REF} ===")
print(f"(bit indices = byte*8 + bit; opcode bits 0-12 hold the major opcode)")
print()

# Sort: shape variants first (16816/16832/INVALID), then everything else
def sort_key(mn):
    parts = mn.split(".")
    return (len(parts), mn)

for mn in sorted(canon.keys(), key=sort_key):
    if mn == REF:
        continue
    asm, hx = canon[mn]
    diff = bit_diff(ref_hex, hx)
    diff_str = ",".join(str(b) for b in diff)
    print(f"{mn:55s}  Δ={len(diff):2d}  bits={diff_str}")
