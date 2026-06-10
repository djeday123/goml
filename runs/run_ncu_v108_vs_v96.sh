#!/bin/bash
# NCu compare v108 step 1 (warp-spec Br=64 2P+2C) vs v96 on PEAK cfg=9.
# CRITICAL diagnostic: did wait stall drop? (means producer overlap works)

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
launch__block_size
launch__grid_size
launch__registers_per_thread
launch__shared_mem_per_block_dynamic
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
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
    local out="$GOML/runs/ncu_${label}_v108cmp.csv"
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
for k in ['launch__block_size','launch__grid_size','launch__registers_per_thread',
          'launch__shared_mem_per_block_dynamic',
          'launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem',
          'sm__warps_active.avg.pct_of_peak_sustained_active',
          'sm__cycles_active.avg.pct_of_peak_sustained_elapsed']:
    if k in launch_metrics:
        print(f"  {k.replace('launch__','').replace('sm__',''):<48} {launch_metrics[k]:>10.2f}")
print()
print("--- STALLS ---")
total = 0
for k,v in sorted(stall_metrics.items(), key=lambda x:-x[1]):
    if v > 0.01:
        print(f"  {k:<24} {v:>7.2f}%")
        total += v
print(f"  {'TOTAL':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

profile "v96"  "$GOML/runs/fa_v96_ksbatched"     9
profile "v108" "$GOML/runs/fa_v108_warpspec_step1" 9

echo "================================================================"
echo "DIAGNOSTIC GUIDE for step 1:"
echo ""
echo "  wait%       v96=37.77   v108=?"
echo "    if v108 wait <30  → overlap WORKING, producer hides K/V latency"
echo "                       → step 2: deeper buffer to amplify"
echo "    if v108 wait ~35  → small overlap, marginal"
echo "    if v108 wait >40  → overlap NOT working (or fewer warps hurt hiding)"
echo ""
echo "  selected%   higher = warps are sitting around waiting to be picked"
echo "  not_selected% = wait-for-pick (occupancy-bound)"
echo ""
echo "  cycles_active = SM time utilization"
echo "  warps_active = avg active warps / max"
echo ""
echo "If perf bad but wait good → bottleneck is downstream (MMA pipe, SMEM port)"
echo "If wait still high → producer overlap didn't materialize"
echo "================================================================"
