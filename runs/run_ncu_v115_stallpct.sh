#!/bin/bash
# NCu stall composition for v115 SHFL transpose — explain -28% perf vs v96 (568T).
# Hypothesis: math_pipe_throttle / short_scb / mio rise because SHFL+byte
# permutation pumps work into MMA/MIO pipes that are already at floor on hd=128.
# UncoalescedShared confirmed dropped 40.0% → 34.7% (only -5pp, not enough).

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v115_shfl_transpose"

cd "$GOML"

if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not built" >&2; exit 1; fi

METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct
smsp__warp_issue_stalled_drain_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
smsp__warp_issue_stalled_misc_per_warp_active.pct
smsp__warp_issue_stalled_sleeping_per_warp_active.pct
smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct
smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct
EOF
)

profile_cfg() {
    local label="$1" cfg="$2"
    local out="$GOML/runs/ncu_v115_stallpct_cfg${cfg}.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$BIN" --ncu "$cfg" > "$out" 2>&1
    python3 << PYEOF
import csv, io
with open("$out") as f:
    lines = f.readlines()
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'):
        hi = i; break
if hi is None:
    print("  ERROR: header not found"); raise SystemExit(0)
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
m = []
for r in rdr:
    n = r.get('Metric Name','')
    if 'stalled' in n.lower() and 'per_warp_active' in n.lower():
        try: v = float(r.get('Metric Value','0').replace(',','.'))
        except: v = 0
        short = n.replace('smsp__warp_issue_stalled_','').replace('_per_warp_active.pct','')
        m.append((short, v))
m.sort(key=lambda x: x[1], reverse=True)
total = sum(v for _,v in m)
print(f"  {'Stall reason':<24} {'%':>8}")
for n,v in m:
    print(f"  {n:<24} {v:>7.2f}%")
print(f"  {'TOTAL stalled':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

profile_cfg "LARGE  bh=64 sl=8192 (393T) ← v115 PEAK"  9

echo "================================================================"
echo "REFERENCE v96 hd=128 PEAK bh=64 sl=8192 (568T):"
echo "  wait                37.77%"
echo "  math_pipe_throttle   8.86%"
echo "  short_scoreboard     6.50%  (approx)"
echo "  mio_throttle         4-5%"
echo "  barrier              4-5%"
echo "  lg_throttle          3.31%"
echo "  Eligible            ~33%"
echo "================================================================"
echo "WATCH FOR v115 vs v96:"
echo "  - math_pipe_throttle DELTA (SHFL + byte ops in MMA pipe?)"
echo "  - short_scoreboard / mio DELTA (LDS.32 instead of 4×LDS.U8?)"
echo "  - barrier DELTA (transpose_v takes longer per iter → idle warps?)"
echo "================================================================"
