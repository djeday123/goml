#!/usr/bin/env python3
"""v96b NCu deep parse — methodology disclosure + top-20 stall table.

METHODOLOGY:
- Classification: by SASS OPCODE (first token of instruction after predicate strip).
- Sample column: 'Warp Stall Sampling (All Samples)' = column 2 — counts ALL warp-cycle
  PC samples (both issued and not-issued). Represents WHERE warp-time is spent.
- Stall reason columns: NCu emits two parallel sets — 'Not-issued' (waiting) and
  'Selected' (issued at this PC). We extract BOTH for the top instructions.
"""
import re, collections

SRC = "/data/lib/podman-data/projects/goml/runs/v96b_pcsamp_source.txt"

with open(SRC) as f:
    lines = f.readlines()

# Locate header dash + Address row to identify columns
dash_line = None; addr_line_idx = None
for i, l in enumerate(lines):
    if l.startswith("----") and i+1 < len(lines) and "Address" in lines[i+1]:
        dash_line = l; addr_line_idx = i+1; break

col_starts = []
in_col = False
for i, c in enumerate(dash_line):
    if c == '-' and not in_col:
        col_starts.append(i); in_col = True
    elif c == ' ':
        in_col = False
col_ends = [s-1 for s in col_starts[1:]] + [len(dash_line)]

# Combine header lines to get column NAMES
def header_text(ci):
    h = ''
    for hl in lines[addr_line_idx:addr_line_idx+10]:
        if hl.startswith('0x') or hl.startswith('----'): break
        h += hl[col_starts[ci]:col_ends[ci]+1].strip() + ' '
    return h.strip()

# Find data start
data_start = None
for i, l in enumerate(lines):
    if l.startswith("0x") and i > addr_line_idx:
        data_start = i; break

# Identify columns by name
COL_ADDR = 0
COL_SASS = 1
COL_ALL_SAMPLES = 2
COL_NOT_ISSUED = 3
COL_HASH_SAMPLES = 4
COL_INSTR_EXEC = 5

# Find stall reason columns by inspecting header
# 16 stall reasons in Not-Issued order, then 16 in Selected order
# Order: barrier, branch_resolving, dispatch_stall, drain, lg_throttle,
#        long_scoreboard, math_pipe_throttle, membar, mio_throttle, misc,
#        no_instruction, not_selected, selected, short_scoreboard, sleeping,
#        tex_throttle, wait
STALL_NAMES = ['barrier','branch','dispatch','drain','lg','long_sb','math',
               'membar','mio','misc','no_instr','not_selected','selected',
               'short_sb','sleeping','tex','wait']
N_STALL = len(STALL_NAMES)  # 17 items

# Locate stall columns: scan column names
stall_not_issued_cols = []
stall_selected_cols = []
for ci in range(7, len(col_starts)):
    name = header_text(ci).lower()
    if 'stall_' in name or any(s in name for s in ['barrier','wait','select','dispat','long','short_s','mio','math','no_in']):
        # crude detection — refine below
        stall_not_issued_cols.append(ci) if len(stall_not_issued_cols) < N_STALL else stall_selected_cols.append(ci)

# Actually safer: from the dump, after col 30 we have ~16-17 Not-Issued cols, then ~16-17 Selected cols
# The exact column count is 65 in our case. Let me hard-code by inspection:
# Cols 31..47 = Not_Issued stalls (17 cols)
# Cols 48..64 = Selected stalls (17 cols)

# Override based on actual count
total_cols = len(col_starts)
# Typical NCu layout: ~30 metric cols + 2*17=34 stall cols = ~64-65 total
if total_cols >= 64:
    stall_not_issued_cols = list(range(total_cols - 34, total_cols - 17))
    stall_selected_cols = list(range(total_cols - 17, total_cols))

def parse_int(s):
    s = s.replace(',', '').strip()
    if not s or s == '-': return 0
    try: return int(s)
    except: return 0

def classify(sass):
    sass = sass.strip()
    if sass.startswith("@"):
        parts = sass.split(None, 1)
        if len(parts) > 1: sass = parts[1]
    op = sass.split(None, 1)[0] if sass else ""
    if op.startswith("HMMA") or op.startswith("QMMA") or op[:3] == "MMA":
        return "MMA"
    if op.startswith("LDS"): return "LDS"
    if op.startswith("STS"): return "STS"
    if op.startswith("LDSM"): return "LDSM"
    if op in ("FFMA","FADD","FMUL","FMNMX","HFMA2","HMUL2","HADD2","FMNMX2"): return "SOFTMAX_MATH"
    if op.startswith("MUFU"): return "MUFU"
    if op in ("F2F","F2I","I2F","FCMP","HABS2","HMNMX2","FSEL","FRND","FCHK","F2FP"): return "CVT_MATH"
    if op.startswith("F2FP"): return "CVT_MATH"
    if op in ("CVT",): return "CVT_MATH"
    if op in ("BAR","MEMBAR","DEPBAR","BARRIER","BSSY","BSYNC"): return "BAR"
    if op.startswith("LDGSTS") or op.startswith("CP") or op == "LDGDEPBAR": return "CPASYNC"
    if op in ("LDG","STG"): return "LDG_STG"
    if op.startswith("SHFL"): return "SHFL"
    if op in ("BRA","BREAK","CALL","RET","EXIT","JMP","JCALL","KILL","BSSY","BSYNC"): return "BRANCH"
    if op.startswith(("IADD","IMAD","LOP","SHF","MOV","LEA","ISETP","S2R","PLOP",
                      "ULD","URD","USHF","UIMAD","UISETP","UIADD","ULOP","UPLOP",
                      "UMOV","ULEA","R2UR","UF2I","UI2F","UFCMP","USEL","UWARP",
                      "ULDC","LDC","LDCU","S2UR","FSETP","HSETP","IABS","PRMT","SEL")):
        return "ADDR_ARITH"
    if op in ("NOP","YIELD"): return "OTHER"
    return f"UNK:{op}"

# Parse all rows
rows = []
for l in lines[data_start:]:
    if not l.startswith("0x"): continue
    addr = l[col_starts[COL_ADDR]:col_ends[COL_ADDR]+1].strip()
    sass = l[col_starts[COL_SASS]:col_ends[COL_SASS]+1].strip()
    all_s = parse_int(l[col_starts[COL_ALL_SAMPLES]:col_ends[COL_ALL_SAMPLES]+1])
    not_iss = parse_int(l[col_starts[COL_NOT_ISSUED]:col_ends[COL_NOT_ISSUED]+1])
    stalls_ni = [parse_int(l[col_starts[c]:col_ends[c]+1]) for c in stall_not_issued_cols]
    stalls_sel = [parse_int(l[col_starts[c]:col_ends[c]+1]) for c in stall_selected_cols]
    cls = classify(sass)
    rows.append({
        'addr': addr, 'sass': sass, 'all': all_s, 'not_iss': not_iss,
        'stalls_ni': stalls_ni, 'stalls_sel': stalls_sel, 'class': cls,
    })

total_all = sum(r['all'] for r in rows)
print(f"=== METHODOLOGY ===")
print(f"Source: ncu --page source --print-source sass on ncu_v96b_baseline.ncu-rep")
print(f"Classification field: OPCODE (first token of SASS after predicate strip)")
print(f"Sample column: 'Warp Stall Sampling (All Samples)' = both issued+stalled")
print(f"Stall breakdown: 17 Not-Issued cols + 17 Selected cols, by reason")
print(f"Total samples: {total_all:,}")
print()

# ===== Top 20 instructions by samples =====
print("=== TOP-20 INSTRUCTIONS BY ALL SAMPLES ===")
print(f"Top stall reasons per row: wait_NI, selected_SEL, short_sb_NI, mio_NI")
print()
rows.sort(key=lambda r: -r['all'])
hdr = f"{'#':<3} {'Addr (last 4 hex)':<8} {'Class':<10} {'Opcode':<10} {'Samples':>8} {'%':>6} {'wait_NI':>8} {'selected':>8} {'short_sb':>8} {'mio':>6}"
print(hdr); print("-" * len(hdr))
for i, r in enumerate(rows[:20]):
    addr_short = r['addr'][-6:]
    opcode = r['sass'].split(None, 1)[0] if r['sass'] else ''
    if opcode.startswith('@'):
        parts = r['sass'].split(None, 2)
        opcode = parts[1] if len(parts) > 1 else opcode
    samples = r['all']
    pct = 100*samples/total_all
    # indices: wait=16 (NI), selected=12 (SEL), short_sb=13 (NI), mio=8 (NI)
    wait_ni = r['stalls_ni'][16] if len(r['stalls_ni']) > 16 else 0
    sel = r['stalls_sel'][12] if len(r['stalls_sel']) > 12 else 0
    short_sb = r['stalls_ni'][13] if len(r['stalls_ni']) > 13 else 0
    mio = r['stalls_ni'][8] if len(r['stalls_ni']) > 8 else 0
    print(f"{i+1:<3} {addr_short:<8} {r['class']:<10} {opcode[:10]:<10} {samples:>8,} {pct:>5.2f}% {wait_ni:>8,} {sel:>8,} {short_sb:>8,} {mio:>6,}")

# ===== MMA class breakdown: selected vs wait =====
print()
print("=== MMA CLASS — SELECTED vs WAIT BREAKDOWN ===")
mma_rows = [r for r in rows if r['class'] == 'MMA']
mma_total = sum(r['all'] for r in mma_rows)
mma_sel = sum(r['stalls_sel'][12] for r in mma_rows if len(r['stalls_sel']) > 12)
mma_wait_ni = sum(r['stalls_ni'][16] for r in mma_rows if len(r['stalls_ni']) > 16)
mma_short_sb = sum(r['stalls_ni'][13] for r in mma_rows if len(r['stalls_ni']) > 13)
mma_mio = sum(r['stalls_ni'][8] for r in mma_rows if len(r['stalls_ni']) > 8)
mma_long_sb = sum(r['stalls_ni'][5] for r in mma_rows if len(r['stalls_ni']) > 5)
mma_math = sum(r['stalls_ni'][6] for r in mma_rows if len(r['stalls_ni']) > 6)
print(f"MMA total samples:       {mma_total:>10,}  ({100*mma_total/total_all:5.2f}% of all)")
print(f"  - selected (issued):   {mma_sel:>10,}  ({100*mma_sel/mma_total:5.2f}% of MMA)")
print(f"  - wait (NI):           {mma_wait_ni:>10,}  ({100*mma_wait_ni/mma_total:5.2f}% of MMA)")
print(f"  - math_pipe (NI):      {mma_math:>10,}  ({100*mma_math/mma_total:5.2f}% of MMA)")
print(f"  - short_sb (NI):       {mma_short_sb:>10,}  ({100*mma_short_sb/mma_total:5.2f}% of MMA)")
print(f"  - long_sb (NI):        {mma_long_sb:>10,}  ({100*mma_long_sb/mma_total:5.2f}% of MMA)")
print(f"  - mio (NI):            {mma_mio:>10,}  ({100*mma_mio/mma_total:5.2f}% of MMA)")
sel_pct_of_total = 100 * mma_sel / total_all
wait_pct_of_total = 100 * mma_wait_ni / total_all
print()
print(f"Of 26.43% MMA share of total:")
print(f"  REAL dispatch (selected):  {sel_pct_of_total:.2f}% of total")
print(f"  Waiting (wait stall on MMA PC):  {wait_pct_of_total:.2f}% of total")
print(f"  Other stalls on MMA PC:    {26.43 - sel_pct_of_total - wait_pct_of_total:.2f}% of total")
