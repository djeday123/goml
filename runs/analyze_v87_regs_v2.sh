#!/bin/bash
# v2: Classify registers by SASS instruction patterns, not source-line mapping.
# Identifies hot registers (used in HMMA, the kv-loop core math) vs cold (only address calc).
set -uo pipefail

SASS_LB3="/data/lib/podman-data/projects/goml/runs/sass_v87_lb3_only.txt"

if [ ! -f "$SASS_LB3" ]; then echo "ERROR: $SASS_LB3 missing — run analyze_v87_regs.sh first" >&2; exit 1; fi

python3 << PYEOF
import re
from collections import defaultdict

# Parse SASS instructions
inst_pat = re.compile(r'^\s*/\*([0-9a-fA-F]+)\*/\s+(\S.*?)(?:\s*;|\s*/\*)')
reg_pat = re.compile(r'\bR(\d+)\b')

# Classify instructions by mnemonic family
def classify_inst(mnem):
    if 'HMMA' in mnem or 'QMMA' in mnem or 'MMA.' in mnem:
        return 'MMA'
    if mnem.startswith('LDS') or mnem.startswith('STS'):
        return 'SMEM_LD_ST'
    if mnem.startswith('LDG') or mnem.startswith('STG') or mnem.startswith('LDGSTS'):
        return 'GMEM_LD_ST'
    if mnem.startswith('LDC') or mnem.startswith('LDCU') or mnem.startswith('UMOV') or mnem.startswith('S2UR'):
        return 'CONST_OR_UNI'
    if mnem.startswith('FFMA') or mnem.startswith('FADD') or mnem.startswith('FMUL') or mnem.startswith('HFMA') or mnem.startswith('HMUL') or mnem.startswith('HADD'):
        return 'FPU_MATH'
    if mnem.startswith('IADD') or mnem.startswith('IMAD') or mnem.startswith('ISETP') or mnem.startswith('IMNMX') or mnem.startswith('LEA') or mnem.startswith('SHF'):
        return 'INT_ARITH'
    if mnem.startswith('BRA') or mnem.startswith('BSSY') or mnem.startswith('BSYNC') or mnem.startswith('EXIT'):
        return 'CONTROL'
    if mnem.startswith('SHFL') or mnem.startswith('BAR') or mnem.startswith('CCTL'):
        return 'WARP_BAR'
    return 'OTHER'

# Track per-register: instruction count per class + which instruction addresses
reg_class_count = defaultdict(lambda: defaultdict(int))  # reg → class → count
reg_first_addr = {}
reg_last_addr = {}
class_count = defaultdict(int)

# Track first MMA and last MMA addresses (kv-loop bounds approximation)
first_mma_addr = None
last_mma_addr = None

with open("$SASS_LB3") as f:
    for raw in f:
        m = inst_pat.search(raw)
        if not m:
            continue
        addr_hex = m.group(1)
        addr = int(addr_hex, 16)
        body = m.group(2).strip()
        # First token is the mnemonic (e.g., HMMA.16832.F8.F8.F16 or IADD3)
        # May have predicate prefix @P0 — strip it
        if body.startswith('@'):
            parts = body.split(None, 2)
            if len(parts) >= 2:
                body = ' '.join(parts[1:])
        first_token = body.split(None, 1)[0]
        cls = classify_inst(first_token)
        class_count[cls] += 1
        if cls == 'MMA':
            if first_mma_addr is None:
                first_mma_addr = addr
            last_mma_addr = addr
        regs = set(int(g) for g in reg_pat.findall(body))
        for r in regs:
            reg_class_count[r][cls] += 1
            if r not in reg_first_addr:
                reg_first_addr[r] = addr
            reg_last_addr[r] = addr

print(f"Total instructions parsed: {sum(class_count.values())}")
print()
print("=== Instruction class breakdown ===")
total = sum(class_count.values())
for cls, c in sorted(class_count.items(), key=lambda x: -x[1]):
    print(f"  {cls:<16} {c:>6}  ({100*c/total:>5.1f}%)")

print()
print(f"=== Approximate kv-loop bounds (first/last MMA addr) ===")
print(f"  First MMA: 0x{first_mma_addr:04x}" if first_mma_addr else "  No MMA found")
print(f"  Last MMA:  0x{last_mma_addr:04x}" if last_mma_addr else "")
print(f"  Loop body span: {(last_mma_addr - first_mma_addr) if first_mma_addr and last_mma_addr else 0} bytes")

print()
print("=== Register classification ===")
all_regs = sorted(reg_class_count.keys())
print(f"Total unique regs: {len(all_regs)} (max R{max(all_regs) if all_regs else 0})")

# Classify each reg:
# HOT_MMA = appears in MMA inst
# HOT_LOOP = appears between first and last MMA but not in MMA
# COLD = appears only OUTSIDE [first_mma, last_mma]
hot_mma = []
hot_loop = []
cold = []
for r in all_regs:
    if reg_class_count[r]['MMA'] > 0:
        hot_mma.append(r)
    elif first_mma_addr is not None and last_mma_addr is not None:
        if reg_first_addr[r] >= first_mma_addr and reg_last_addr[r] <= last_mma_addr:
            hot_loop.append(r)
        elif reg_first_addr[r] < first_mma_addr and reg_last_addr[r] < first_mma_addr:
            cold.append(r)  # pre-loop only
        elif reg_first_addr[r] > last_mma_addr:
            cold.append(r)  # post-loop (writeback) only
        else:
            # spans both pre and loop, or loop and post — partially hot
            hot_loop.append(r)

print(f"  HOT (MMA operand):              {len(hot_mma)} regs")
print(f"  HOT (in loop body, non-MMA):    {len(hot_loop)} regs")
print(f"  COLD (only pre-loop OR post):   {len(cold)} regs")

print()
print(f"=== Cold register IDs (candidates for extract) ===")
print(f"  {sorted(cold)}")

print()
print(f"=== HOT MMA register IDs (untouchable) ===")
print(f"  {sorted(hot_mma)[:50]}")
if len(hot_mma) > 50:
    print(f"  ... and {len(hot_mma) - 50} more")

# Now identify which MMA regs are accumulator (Or_p, Sr_p) vs A/B operands (Qr, Pr, K, V):
# Accumulators appear MORE TIMES in MMA than A/B operands (they're both read and written)
print()
print("=== MMA register intensity (top 40 by MMA-use count) ===")
mma_intensity = [(r, reg_class_count[r]['MMA']) for r in hot_mma]
mma_intensity.sort(key=lambda x: -x[1])
for r, c in mma_intensity[:40]:
    print(f"  R{r:<3}  {c:>4} MMA-uses")

print()
print("=== 4-block plan implications ===")
n_cold = len(cold)
print(f"  v87 LB=3 used 160 regs/thread")
print(f"  Need to free 33 to reach 127 (4-block budget) — or ≤127 regs/thread")
print(f"  Cold reg pool (extract candidates): {n_cold}")
if n_cold >= 33:
    print(f"  → SUFFICIENT cold pool — extraction MAY enable 4 blocks")
    print(f"  → Risk: extraction adds SMEM ops, possible mio_throttle growth (predicted user)")
elif n_cold >= 15:
    print(f"  → MODERATE cold pool — needs combined approach (extraction + recompute)")
else:
    print(f"  → SMALL cold pool ({n_cold}) — most regs are HOT (in kv-loop)")
    print(f"  → 4-block plan structurally hard: little to extract without perf loss")
    print(f"  → Aligns with v88 Or_p result (-83% from extracting hot state)")
PYEOF
