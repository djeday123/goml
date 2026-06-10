#!/bin/bash
# NCu v111 PRODUCER-SKIP vs v110 baseline on PEAK cfg=9.
# Validate that producer-skip transpose actually reduces barrier stall.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
launch__block_size
launch__registers_per_thread
sm__warps_active.avg.pct_of_peak_sustained_active
sm__cycles_active.avg.pct_of_peak_sustained_elapsed
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct
smsp__warp_issue_stalled_drain_per_warp_active.pct
smsp__warp_issue_stalled_not_selected_per_warp_active.pct
smsp__warp_issue_stalled_selected_per_warp_active.pct
EOF
)

profile() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_pskip.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
    python3 << PYEOF
import csv, io
with open("$out") as f: lines = f.readlines()
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'):
        hi = i; break
if hi is None:
    print("  ERROR header"); raise SystemExit(0)
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
launch_metrics, stall_metrics = {}, {}
for r in rdr:
    n = r.get('Metric Name','')
    try: v = float(r.get('Metric Value','0').replace(',','.'))
    except: v = 0
    if n.startswith('launch__') or n.startswith('sm__'):
        launch_metrics[n] = v
    elif 'stalled' in n.lower() and 'per_warp_active' in n.lower():
        short = n.replace('smsp__warp_issue_stalled_','').replace('_per_warp_active.pct','')
        stall_metrics[short] = v
print()
print("--- LAUNCH ---")
for k in ['launch__block_size','launch__registers_per_thread',
          'sm__warps_active.avg.pct_of_peak_sustained_active',
          'sm__cycles_active.avg.pct_of_peak_sustained_elapsed']:
    if k in launch_metrics:
        print(f"  {k.replace('launch__','').replace('sm__',''):<48} {launch_metrics[k]:>10.2f}")
print()
print("--- STALLS (real = total − selected − not_selected) ---")
total = 0
real_total = 0
for k,v in sorted(stall_metrics.items(), key=lambda x:-x[1]):
    if v > 0.01:
        print(f"  {k:<24} {v:>7.2f}%")
        total += v
        if k not in ('selected', 'not_selected'):
            real_total += v
print(f"  {'REAL STALLS':<24} {real_total:>7.2f}%")
print(f"  {'Real Eligible':<24} {100-real_total:>7.2f}%")
PYEOF
    echo ""
}

profile "v110" "$GOML/runs/fa_v110_warpspec_1p3c"   9
profile "v111" "$GOML/runs/fa_v111_producer_skip"   9

echo "================================================================"
echo "v96 ref:  wait 37.78 / barrier 2.00 / Eligible 32.90 / perf 564"
echo "v110:     wait 35.78 / barrier 10.38 / Eligible 28.49 / perf 454"
echo "v111:     ? / ? / ? / 484 mean"
echo ""
echo "Expected: barrier ↓ significantly (producer no longer waits at post-transpose)"
echo "Expected: Eligible ↑ correspondingly"
echo "================================================================"
