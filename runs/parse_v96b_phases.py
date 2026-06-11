#!/usr/bin/env python3
"""Parse v96b NCu source-attributed dump.
Group warp-stall samples by KERNEL PHASE based on SASS instruction families:
- QMMA-block: ks-batched QK and PV MMA chains (HMMA/QMMA family)
- LDS-PV: PV B-operand fetches from smV_T
- LDS-QK: QK B-operand fetches from smK
- STS-smP: P-quantize stores
- FFMA/FADD/FMUL/MUFU: softmax math
- BAR/MEMBAR: synchronization
- LDG/STG: epilogue or initial load
- Other: address arithmetic, branches, etc.
"""
import re, collections

SRC = "/data/lib/podman-data/projects/goml/runs/v96b_pcsamp_source.txt"

# Find header
with open(SRC) as f:
    lines = f.readlines()

# Find dash line near data start
dash_line = None
addr_line_idx = None
for i, l in enumerate(lines):
    if l.startswith("----") and i+1 < len(lines) and "Address" in lines[i+1]:
        dash_line = l
        addr_line_idx = i+1
        break

col_starts = []
in_col = False
for i, c in enumerate(dash_line):
    if c == '-' and not in_col:
        col_starts.append(i); in_col = True
    elif c == ' ':
        in_col = False
col_ends = [s-1 for s in col_starts[1:]] + [len(dash_line)]

# Find data start
data_start = None
for i, l in enumerate(lines):
    if l.startswith("0x") and i > addr_line_idx:
        data_start = i; break

# Column 2 = Warp Stall Sampling (All Samples) — column for total samples
# Column 3 = (Not-issued Samples)
# Column 4 = # Samples (unique PC sampling)
SAMPLES_COL = 2  # Warp Stall Sampling All Samples

# Classify SASS
def classify(sass):
    sass = sass.strip()
    # Strip predicate
    if sass.startswith("@"):
        parts = sass.split(None, 1)
        if len(parts) > 1: sass = parts[1]
    op = sass.split(None, 1)[0] if sass else ""

    # QMMA/HMMA — actual MMA instructions (QK + PV)
    if op.startswith("HMMA") or op.startswith("QMMA") or "MMA" == op[:3]:
        return "MMA"
    # LDS / STS / LDSM — SMEM access (PV V-fetch, QK K-fetch, P-store)
    if op in ("LDS", "LDS.U", "LDS.U.32", "LDS.U.64", "LDS.U.128") or op.startswith("LDS"):
        return "LDS"
    if op in ("STS",) or op.startswith("STS"):
        return "STS"
    if op in ("LDSM",) or op.startswith("LDSM"):
        return "LDSM"
    # Softmax math: FFMA, FADD, FMUL, FMNMX (fmaxf), MUFU (ex2/log), HFMA2, HMUL2, HADD2
    if op in ("FFMA","FADD","FMUL","FMNMX","MUFU","HFMA2","HMUL2","HADD2","FMNMX2"):
        return "SOFTMAX_MATH"
    if op in ("F2F","F2I","I2F","FCMP","HABS2","HMNMX2","FSEL","FRND","FCHK"):
        return "SOFTMAX_MATH"
    # P-quantize / conversion
    if op in ("CVT","HFMA2.MMA"):  # cvt.satfinite.e4m3 etc
        return "CVT"
    # Barriers
    if op in ("BAR","MEMBAR","DEPBAR","BARRIER","BSSY","BSYNC"):
        return "BAR"
    # Global memory (epilogue, init)
    if op in ("LDG","STG"):
        return "LDG_STG"
    if op in ("LDGSTS","CPASYNC") or op.startswith("LDGSTS") or op.startswith("CP"):
        return "CPASYNC"
    # Shuffle (used in reductions)
    if op.startswith("SHFL"):
        return "SHFL"
    # Branch/exit
    if op in ("BRA","BSSY","BREAK","CALL","RET","EXIT","JMP","JCALL","KILL"):
        return "BRANCH"
    # Address arithmetic, integer ops
    if op.startswith("IADD") or op.startswith("IMAD") or op.startswith("LOP") or \
       op.startswith("SHF") or op.startswith("MOV") or op.startswith("LEA") or \
       op.startswith("ISETP") or op.startswith("S2R") or op.startswith("PLOP") or \
       op.startswith("ULD") or op.startswith("URD") or op.startswith("USHF") or \
       op.startswith("UIMAD") or op.startswith("UISETP") or op.startswith("UIADD") or \
       op.startswith("ULOP") or op.startswith("UPLOP") or op.startswith("UMOV") or \
       op.startswith("ULEA") or op.startswith("R2UR") or op.startswith("UF2I") or \
       op.startswith("UI2F") or op.startswith("UISETP") or op.startswith("UFCMP") or \
       op.startswith("USEL") or op.startswith("UWARP") or op.startswith("ULDC") or \
       op.startswith("LDC") or op.startswith("LDCU") or op.startswith("S2UR"):
        return "ADDR_ARITH"
    if op.startswith("FSETP") or op.startswith("HSETP"):
        return "ADDR_ARITH"
    if op == "NOP" or op == "YIELD":
        return "OTHER"
    return f"UNK:{op}"

agg = collections.Counter()
top_per_class = collections.defaultdict(list)
total_samples = 0

for l in lines[data_start:]:
    if not l.startswith("0x"): continue
    addr = l[col_starts[0]:col_ends[0]+1].strip()
    sass = l[col_starts[1]:col_ends[1]+1].strip()
    samples_str = l[col_starts[SAMPLES_COL]:col_ends[SAMPLES_COL]+1].strip()
    try: samples = int(samples_str)
    except: samples = 0
    if samples <= 0: continue
    cls = classify(sass)
    agg[cls] += samples
    total_samples += samples
    top_per_class[cls].append((samples, addr, sass[:80]))

print(f"Total warp-stall samples (All): {total_samples:,}")
print()
print(f"{'Phase class':<20} {'Samples':>12} {'% of total':>10}")
print("=" * 50)
for cls, n in sorted(agg.items(), key=lambda x: -x[1]):
    pct = 100*n/total_samples if total_samples else 0
    print(f"  {cls:<20} {n:>12,} {pct:>9.2f}%")
print()

# Compute MMA vs non-MMA share
mma_share = 100 * agg.get("MMA", 0) / total_samples
non_mma = 100 - mma_share
print(f"MMA share:     {mma_share:.2f}%")
print(f"Non-MMA share: {non_mma:.2f}%")
print()

# Top non-MMA contributors
print("Top non-MMA classes (excluding MMA):")
non_mma_classes = [(c, n) for c, n in agg.items() if c != "MMA"]
non_mma_classes.sort(key=lambda x: -x[1])
non_mma_total = sum(n for _, n in non_mma_classes)
for c, n in non_mma_classes[:6]:
    pct_of_nonmma = 100 * n / non_mma_total if non_mma_total else 0
    pct_of_total  = 100 * n / total_samples
    print(f"  {c:<20} {n:>10,}  {pct_of_total:>6.2f}% of total, {pct_of_nonmma:>6.2f}% of non-MMA")

print()
print("Top 5 non-MMA hot addresses (likely phase concentrators):")
all_non_mma = []
for c in agg:
    if c == "MMA": continue
    for s, a, sass in top_per_class[c]:
        all_non_mma.append((s, a, sass, c))
all_non_mma.sort(reverse=True)
for s, a, sass, c in all_non_mma[:10]:
    print(f"  {s:>6}  [{c:<14}]  {a}  {sass}")
