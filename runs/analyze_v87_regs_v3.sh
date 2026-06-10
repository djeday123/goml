#!/bin/bash
# v3: Gradient register intensity analysis — count all reg references in kv-loop.
# Classify each reg by total uses (read+write) inside loop body, separate MMA vs non-MMA.
# Goal: find "cool" regs (≤3 uses) that may be extractable without performance loss.
set -uo pipefail

SASS_LB3="/data/lib/podman-data/projects/goml/runs/sass_v87_lb3_only.txt"

if [ ! -f "$SASS_LB3" ]; then echo "ERROR: $SASS_LB3 missing — run analyze_v87_regs.sh first" >&2; exit 1; fi

python3 << PYEOF
import re
from collections import defaultdict

inst_pat = re.compile(r'^\s*/\*([0-9a-fA-F]+)\*/\s+(\S.*?)(?:\s*;|\s*/\*)')
reg_pat = re.compile(r'\bR(\d+)\b')

# Per-register tracking:
#   addr_first, addr_last for phase classification
#   in_loop_count (total uses inside kv-loop)
#   out_loop_count (uses outside)
#   mma_count (number of MMA instructions where this reg appears)
reg_in_loop = defaultdict(int)
reg_out_loop = defaultdict(int)
reg_mma_count = defaultdict(int)
reg_first_addr = {}
reg_last_addr = {}

# First/last MMA addr (kv-loop bounds)
first_mma_addr = None
last_mma_addr = None
all_insts = []

# Pass 1: collect instructions and find loop bounds
with open("$SASS_LB3") as f:
    for raw in f:
        m = inst_pat.search(raw)
        if not m:
            continue
        addr = int(m.group(1), 16)
        body = m.group(2).strip()
        if body.startswith('@'):
            parts = body.split(None, 2)
            if len(parts) >= 2:
                body = ' '.join(parts[1:])
        mnem = body.split(None, 1)[0]
        is_mma = ('HMMA' in mnem or 'QMMA' in mnem or 'MMA.' in mnem)
        if is_mma:
            if first_mma_addr is None:
                first_mma_addr = addr
            last_mma_addr = addr
        all_insts.append((addr, body, mnem, is_mma))

# Pass 2: classify each instruction in/out loop, count register uses
for addr, body, mnem, is_mma in all_insts:
    in_loop = (first_mma_addr is not None and last_mma_addr is not None
               and first_mma_addr <= addr <= last_mma_addr)
    regs = set(int(g) for g in reg_pat.findall(body))
    for r in regs:
        if in_loop:
            reg_in_loop[r] += 1
        else:
            reg_out_loop[r] += 1
        if is_mma:
            reg_mma_count[r] += 1
        if r not in reg_first_addr:
            reg_first_addr[r] = addr
        reg_last_addr[r] = addr

all_regs = sorted(set(list(reg_in_loop.keys()) + list(reg_out_loop.keys())))
print(f"Total unique regs: {len(all_regs)}")
print(f"First MMA: 0x{first_mma_addr:04x}, Last MMA: 0x{last_mma_addr:04x}")
print(f"Loop body span: {last_mma_addr - first_mma_addr} bytes")
print()

# Gradient classification by IN-LOOP usage
HOT_THRESHOLD = 10
WARM_THRESHOLD = 4
COOL_THRESHOLD = 3

# Categorize each reg
categories = {'hot': [], 'warm': [], 'cool': [], 'cold': []}
for r in all_regs:
    uses_in = reg_in_loop.get(r, 0)
    uses_out = reg_out_loop.get(r, 0)
    if uses_in == 0:
        categories['cold'].append(r)
    elif uses_in <= COOL_THRESHOLD:
        categories['cool'].append(r)
    elif uses_in < HOT_THRESHOLD:
        categories['warm'].append(r)
    else:
        categories['hot'].append(r)

print("=== Register classification by in-loop intensity ===")
print(f"  HOT  (>10 uses in loop):     {len(categories['hot']):>3} regs  [untouchable]")
print(f"  WARM (4-10 uses in loop):    {len(categories['warm']):>3} regs  [risky to extract]")
print(f"  COOL (1-3 uses in loop):     {len(categories['cool']):>3} regs  [extract candidates]")
print(f"  COLD (0 uses in loop):       {len(categories['cold']):>3} regs  [already-cold, free candidates]")
print()

# Of the COOL regs, separate by whether they appear in MMA
cool_mma = [r for r in categories['cool'] if reg_mma_count.get(r, 0) > 0]
cool_nonmma = [r for r in categories['cool'] if reg_mma_count.get(r, 0) == 0]

print("=== COOL regs (1-3 uses) breakdown ===")
print(f"  COOL with MMA touch ({len(cool_mma)}): {sorted(cool_mma)}")
print(f"  COOL no-MMA      ({len(cool_nonmma)}): {sorted(cool_nonmma)}")
print()

# Detailed table for COOL regs
print("=== COOL regs detail (R# | in-loop | out-loop | MMA-count | type-guess) ===")
def guess_type(r):
    uses_in = reg_in_loop.get(r, 0)
    mma = reg_mma_count.get(r, 0)
    if mma == uses_in and mma == 2:
        return "MMA fragment (D pair? — read+write same MMA)"
    elif mma > 0:
        return "MMA operand (B load or accumulator partial)"
    elif uses_in == 1:
        return "live-once temp (load → use → die)"
    elif uses_in == 2:
        return "load → use, or 2-step compute"
    elif uses_in == 3:
        return "small chain (load+modify+store?)"
    return "unknown"

all_cool = sorted(categories['cool'])
for r in all_cool[:60]:
    uses_in = reg_in_loop.get(r, 0)
    uses_out = reg_out_loop.get(r, 0)
    mma = reg_mma_count.get(r, 0)
    print(f"  R{r:<3}  in={uses_in:<2}  out={uses_out:<2}  mma={mma:<2}  {guess_type(r)}")
if len(all_cool) > 60:
    print(f"  ... +{len(all_cool) - 60} more cool regs")
print()

# Distribution histogram of in-loop usage
print("=== In-loop usage histogram ===")
bins = [(0,0), (1,1), (2,2), (3,3), (4,5), (6,10), (11,20), (21,50), (51,100), (101,9999)]
for lo, hi in bins:
    count = sum(1 for r in all_regs if lo <= reg_in_loop.get(r, 0) <= hi)
    label = f"{lo}-{hi}" if lo != hi else str(lo)
    print(f"  uses {label:<6}: {count:>3} regs")
print()

# Final extractable pool: cold + cool-non-mma + cool-mma-with-≤2-uses-and-not-D-output
extractable_easy = categories['cold'] + cool_nonmma
extractable_with_recompute = [r for r in cool_mma if reg_in_loop[r] <= 2]
total_extractable = sorted(set(extractable_easy + extractable_with_recompute))

print("=== Final extractable pool analysis ===")
print(f"  Easy extracts (cold + cool-no-MMA): {len(extractable_easy)}")
print(f"  With recompute (cool-MMA ≤2 uses): {len(extractable_with_recompute)}")
print(f"  TOTAL potential extract pool:       {len(total_extractable)}")
print()

n_total = len(total_extractable)
print(f"=== 4-block plan re-assessment ===")
print(f"  v87 LB=3 used 160 regs, need ≤127 for 4 blocks (free 33)")
print(f"  Extractable pool (cold + cool): {n_total}")
if n_total >= 33:
    print(f"  → SUFFICIENT pool — gradient analysis OPENS path that binary HOT/COLD missed")
    print(f"  → Next: identify which cool regs are concrete .cu variables (Pf_pair? Pr0/1? indices?)")
    print(f"  → Risk: each extraction adds SMEM ops; cumulative mio impact uncertain")
elif n_total >= 20:
    print(f"  → MODERATE — pool is sizable but not full 33. Combined with recompute may work.")
elif n_total >= 11:
    print(f"  → MARGINAL — only {n_total - 11} more than original 11 cold count.")
    print(f"  → 4-block plan still structurally hard but slightly less closed.")
else:
    print(f"  → CLOSED — gradient analysis confirms binary HOT/COLD verdict.")
PYEOF
