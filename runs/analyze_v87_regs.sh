#!/bin/bash
# Static SASS analysis of v87 LB=3 register usage by phase.
# Identifies cold registers — used only outside the hot kv-loop (pre-loop, writeback).
# Cold regs are candidates for SMEM extraction or recompute without performance loss.
set -uo pipefail

CUOBJ=/usr/local/cuda-13.1/bin/cuobjdump
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v87_ksbatched_fp8"
SASS="$GOML/runs/sass_v87_lb3.txt"
SRC="$GOML/libs/flash_attention_v87_hd64_ksbatched_fp8_forward.cu"

cd "$GOML"

if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not found — build v87 first" >&2; exit 1; fi
if [ ! -x "$CUOBJ" ]; then echo "ERROR: cuobjdump not at $CUOBJ" >&2; exit 1; fi

echo "=== Step 1: dump SASS of v87 LB=3 ==="
"$CUOBJ" --dump-sass "$BIN" > "$SASS" 2>&1
echo "wrote $SASS ($(wc -l < "$SASS") lines)"

# Find the LB=3 function block in SASS
echo ""
echo "=== Functions in binary ==="
grep -E "^\s*Function : _Z11fa87_kernel" "$SASS" | head -5

LB3_FUNC="_Z11fa87_kernelILi3EEvPKhS1_S1_P6__halfiiifffi"
echo ""
echo "=== Step 2: extract LB=3 kernel SASS only ==="
SASS_LB3="$GOML/runs/sass_v87_lb3_only.txt"
python3 << PYEOF
import re
with open("$SASS") as f: content = f.read()
# Find start of LB=3 function
fn_pat = re.compile(r"^\s*Function : ${LB3_FUNC}", re.MULTILINE)
m = fn_pat.search(content)
if not m:
    print("ERROR: could not find LB=3 function in SASS")
    raise SystemExit(0)
start = m.start()
# Find end (next "Function :" or end of file)
next_fn = re.search(r"^\s*Function : ", content[m.end():], re.MULTILINE)
end = m.end() + next_fn.start() if next_fn else len(content)
with open("$SASS_LB3", 'w') as f: f.write(content[start:end])
lines = content[start:end].count('\n')
print(f"wrote $SASS_LB3 ({lines} lines)")
PYEOF

echo ""
echo "=== Step 3: register usage per source-line range ==="
python3 << PYEOF
import re
from collections import defaultdict

# Source line ranges of v87 phases (from manual inspection of v87 source)
# Each range: (name, start_line, end_line)
PHASES = [
    ("pre_loop_setup",          1, 295),    # kernel start to kv loop start
    ("kv_loop_K_V_wait_sync",   296, 320),  # cpa_wait + sync
    ("kv_loop_transpose_v",     321, 330),  # transpose_v call
    ("kv_loop_K_prefetch",      331, 350),  # next K prefetch
    ("QK_MMA_ks0_batch",        331, 380),  # explicit ks=0 (v87 reorder)
    ("QK_MMA_ks1_batch",        381, 420),  # explicit ks=1
    ("softmax_phase_BCD",       421, 475),  # rmax + rscale + Or_p rescale
    ("softmax_phase_E",         476, 515),  # ex2 + ns
    ("smP_quantize_write",      516, 540),  # smP write loop
    ("PV_MMA_ks0_batch",        541, 595),  # PV ks=0
    ("PV_MMA_ks1_batch",        596, 645),  # PV ks=1
    ("writeback_loop",          646, 700),  # final O write
]

# Parse SASS: each instruction has format like:
# /*0050*/  IADD3 R5, R6, R7, RZ ;
# Or with .lineinfo:
# //## File "/path/source.cu", line 295
# Then instructions following that line annotation belong to that source line.

reg_pat = re.compile(r'\bR(\d+)\b')
line_pat = re.compile(r'//##\s*File\s+"[^"]*",\s*line\s+(\d+)')
inst_pat = re.compile(r'^\s*/\*[0-9a-fA-F]+\*/\s+(\S.*?)\s*;')

current_line = 0
# reg → phase → count of instructions referencing it
reg_phase = defaultdict(lambda: defaultdict(int))
# phase → set of regs used
phase_regs = defaultdict(set)
# instruction count per phase
phase_inst_count = defaultdict(int)

def phase_for_line(line):
    for name, start, end in PHASES:
        if start <= line <= end:
            return name
    return "other"

with open("$SASS_LB3") as f:
    for raw in f:
        line_match = line_pat.search(raw)
        if line_match:
            current_line = int(line_match.group(1))
            continue
        inst_match = inst_pat.search(raw)
        if not inst_match:
            continue
        phase = phase_for_line(current_line)
        regs_in_inst = set(int(g) for g in reg_pat.findall(inst_match.group(1)))
        phase_inst_count[phase] += 1
        for r in regs_in_inst:
            reg_phase[r][phase] += 1
            phase_regs[phase].add(r)

# Print summary
print()
print("=== Phase instruction counts ===")
total_inst = sum(phase_inst_count.values())
for name, _, _ in PHASES + [("other", 0, 0)]:
    if name in phase_inst_count:
        c = phase_inst_count[name]
        print(f"  {name:<28} {c:>6} insts  ({100*c/max(total_inst,1):>5.1f}%)")
print(f"  {'TOTAL':<28} {total_inst:>6} insts")

print()
print("=== Register count per phase ===")
for name, _, _ in PHASES + [("other", 0, 0)]:
    if name in phase_regs:
        print(f"  {name:<28} {len(phase_regs[name]):>3} unique regs")

print()
print("=== HOT vs COLD register classification ===")
HOT_PHASES = {
    "kv_loop_K_V_wait_sync", "kv_loop_transpose_v", "kv_loop_K_prefetch",
    "QK_MMA_ks0_batch", "QK_MMA_ks1_batch",
    "softmax_phase_BCD", "softmax_phase_E", "smP_quantize_write",
    "PV_MMA_ks0_batch", "PV_MMA_ks1_batch",
}
COLD_PHASES = {"pre_loop_setup", "writeback_loop"}

cold_only_regs = []  # used in cold phases AND NOT in hot phases
hot_regs = set()     # touched by any hot phase
for r, phases in reg_phase.items():
    if any(p in HOT_PHASES for p in phases):
        hot_regs.add(r)
    elif any(p in COLD_PHASES for p in phases):
        cold_only_regs.append(r)

all_regs = sorted(reg_phase.keys())
print(f"  Total unique regs: {len(all_regs)} (max R{max(all_regs) if all_regs else 0})")
print(f"  HOT regs (used in kv-loop body):     {len(hot_regs)}")
print(f"  COLD regs (used only outside loop):  {len(cold_only_regs)}")
if cold_only_regs:
    print(f"  Cold reg IDs: {sorted(cold_only_regs)[:30]}{'...' if len(cold_only_regs) > 30 else ''}")

print()
print("=== Top 30 hottest registers (most used in kv-loop) ===")
hot_usage = []
for r in hot_regs:
    total_hot = sum(reg_phase[r][p] for p in HOT_PHASES if p in reg_phase[r])
    hot_usage.append((r, total_hot))
hot_usage.sort(key=lambda x: x[1], reverse=True)
for r, c in hot_usage[:30]:
    # Show which phases this reg appears in
    phases_used = [p for p in reg_phase[r] if reg_phase[r][p] > 0]
    print(f"  R{r:<3}  {c:>4} hot-uses  in: {','.join(sorted(phases_used)[:5])}")

print()
print("=== Implications for 4-block plan ===")
n_cold = len(cold_only_regs)
print(f"  Need to free 33 regs from v87's 160 to reach 127 (4-block budget)")
print(f"  Cold regs available: {n_cold}")
if n_cold >= 33:
    print(f"  → Cold reg pool is LARGE — extracting cold regs to SMEM/recompute MAY work")
    print(f"  → Even partial extraction (e.g. 15-20 regs) approaches the budget")
elif n_cold >= 15:
    print(f"  → Cold reg pool is MODERATE — extraction gives partial gain, need")
    print(f"    additional levers (recompute, simpler softmax state) to reach 127")
else:
    print(f"  → Cold reg pool is SMALL — most regs are HOT (in kv-loop)")
    print(f"  → 4-block plan is STRUCTURALLY HARD: nothing to extract without perf loss")
    print(f"  → Confirms v88 Or_p result: extracting hot state causes mio explosion")
PYEOF
