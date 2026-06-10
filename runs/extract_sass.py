#!/usr/bin/env python3
"""
Extract SASS mnemonics from sm120_cache.txt produced by nv_isa_solver.

Cache line format:  <asm> --- <hex>
We pull the leading mnemonic token (strip optional @P<n>/@!P<n>/@U... predicate)
and count occurrences. The result is a clean catalog of opcodes nvdisasm
will emit for sm_120a.
"""
import re
import sys
from collections import Counter, defaultdict

CACHE = "/data/lib/podman-data/projects/goml/runs/sm120_cache.txt"
OUT_FULL = "/data/lib/podman-data/projects/goml/runs/sass_catalog.txt"
OUT_TENSOR = "/data/lib/podman-data/projects/goml/runs/sass_tensor.txt"
# Tensor-core / MMA families we care about for our experiment
TENSOR_FAMS = {"HMMA", "QMMA", "OMMA", "IMMA", "DMMA", "BMMA",
               "LDSM", "LDGSTS", "STGM",  # data-movement for MMA pipelines
               "MUFU", "WGMMA", "TCGEN05", "TMA",
               "UTMALDG", "UTMAREDG", "UTMASTG", "UTMAPF"}

# Strip leading predicate tokens like @P0, @!P0, @UP0, @!UP0, @PT
PRED_RE = re.compile(r"^@!?U?P[T0-9]*\s+")
# A real SASS mnemonic: uppercase letters/digits/dots, no embedded spaces
MNEM_RE = re.compile(r"^[A-Z][A-Z0-9_.]*$")

counts = Counter()
# Map "base" mnemonic (first token before the first '.') to set of full variants
families = defaultdict(set)

with open(CACHE) as f:
    for line in f:
        if "---" not in line:
            continue
        asm = line.split("---", 1)[0].strip()
        if not asm:
            continue
        # Drop predicate
        asm = PRED_RE.sub("", asm).strip()
        if not asm:
            continue
        # First whitespace-delimited token is the mnemonic
        tok = asm.split()[0].rstrip(";,")
        # Filter out tokens that are clearly not opcodes (register / immediate fragments)
        if not MNEM_RE.match(tok):
            continue
        counts[tok] += 1
        base = tok.split(".", 1)[0]
        families[base].add(tok)

def render_all(fh):
    fh.write(f"Total unique opcode variants: {len(counts)}\n")
    fh.write(f"Total unique opcode families: {len(families)}\n\n")
    fh.write("=== Top families (by total variant count) ===\n")
    fam_count = sorted(families.items(), key=lambda kv: -len(kv[1]))
    for base, variants in fam_count:
        fh.write(f"  {base:12s}  {len(variants):4d} variants\n")
    fh.write("\n=== All families with their variants ===\n")
    for base in sorted(families):
        variants = sorted(families[base])
        fh.write(f"\n[{base}]  ({len(variants)} variants)\n")
        for v in variants:
            fh.write(f"  {counts[v]:6d}  {v}\n")

def render_tensor(fh):
    fh.write("# Tensor-core / TC-related SASS families on sm_120a\n")
    fh.write("# (extracted from nv_isa_solver cache, 503 unique instruction signatures)\n\n")
    found = [b for b in TENSOR_FAMS if b in families]
    missing = sorted(TENSOR_FAMS - set(families))
    fh.write(f"Found {len(found)} of {len(TENSOR_FAMS)} tracked families. "
             f"Missing on sm_120a: {missing}\n\n")
    for base in sorted(found):
        variants = sorted(families[base])
        fh.write(f"[{base}]  ({len(variants)} variants)\n")
        for v in variants:
            fh.write(f"  {counts[v]:6d}  {v}\n")
        fh.write("\n")

with open(OUT_FULL, "w") as fh:
    render_all(fh)
with open(OUT_TENSOR, "w") as fh:
    render_tensor(fh)
print(f"Wrote {OUT_FULL}")
print(f"Wrote {OUT_TENSOR}")
print(f"Variants: {len(counts)}, families: {len(families)}")
