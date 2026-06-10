#!/bin/bash
# NCu stall % composition for v85 (ILP softmax Phase E refactor).
# Compare WIN LB=3 vs v81 baseline (wait=28.40%, total stalled=64.27%).
# If v85 wait < 28% → Phase E ILP helped, dig deeper.
# If v85 wait ≈ 28% → Phase E wasn't the source, move to other levers.
set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v85_ilpsoftmax_fp8"

cd "$GOML"

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
    local label="$1" cfg="$2" lb="$3"
    local out="$GOML/runs/ncu_v85_stallpct_cfg${cfg}_lb${lb}.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg LB=$lb"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$BIN" --ncu "$cfg" "$lb" > "$out" 2>&1
    echo "wrote $out"

    python3 << PYEOF
import csv, io
with open("$out") as f:
    lines = f.readlines()
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'):
        hi = i; break
if hi is None:
    print("  ERROR: header not found")
    raise SystemExit(0)
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
print(f"  {'TOTAL':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

profile_cfg "v85 WIN LB=3 (390T, +1.3% on bh=8 sl=8192)" 6 3
profile_cfg "v85 WIN LB=2 (baseline LB=2)" 6 2

echo "================================================================"
echo "v81 baseline reference (run_ncu_v81_stallpct results):"
echo "  WIN LB=3: wait 28.40%, math_pipe 10.61%, short_scb 6.63%, TOTAL 64.27%"
echo "  WIN LB=2: wait 35.74%, short_scb 8.59%,  math_pipe 6.80%,  TOTAL 68.52%"
echo ""
echo "v85 wait stall change tells us:"
echo "  < 25%  → Phase E ILP attacked wait → continue ILP in Phase B/D/MMA"
echo "  ≈ 28%  → Phase E not the source → try ILP in Phase D (Or_p rescale)"
echo "             or MMA chain ILP (multiple accumulators)"
echo "  > 30%  → ILP refactor HURT scheduling → revert"
echo "================================================================"
