#!/bin/bash
# NCu compare v96 vs v106 stalls + occupancy on PEAK (cfg=9 = bh=64 sl=8192).
# Goal: understand WHY v106 (12 warps/SM × 2 blocks) is -15.6% vs v96 (8 warps/SM × 2 blocks).
# Hypothesis from v102 occupancy-bottleneck-shift: more warps → wait drops, but math_pipe
# + short_scb + mio rise. Verify same mechanism on v106.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
launch__block_size
launch__registers_per_thread
launch__shared_mem_per_block_dynamic
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct
EOF
)

profile() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_v106cmp.csv"
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
    if n.startswith('launch__') or n.startswith('sm__warps_active'):
        launch_metrics[n] = v
    elif 'stalled' in n.lower() and 'per_warp_active' in n.lower():
        short = n.replace('smsp__warp_issue_stalled_','').replace('_per_warp_active.pct','')
        stall_metrics[short] = v
print()
print("--- OCCUPANCY ---")
for k in ['launch__block_size','launch__registers_per_thread','launch__shared_mem_per_block_dynamic',
          'launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem',
          'sm__warps_active.avg.pct_of_peak_sustained_active']:
    if k in launch_metrics:
        print(f"  {k.replace('launch__','').replace('sm__','sm__'):<48} {launch_metrics[k]:>10.2f}")
print()
print("--- STALLS ---")
total = 0
for k,v in sorted(stall_metrics.items(), key=lambda x:-x[1]):
    print(f"  {k:<24} {v:>7.2f}%")
    total += v
print(f"  {'TOTAL':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

profile "v96"  "$GOML/runs/fa_v96_ksbatched"  9
profile "v106" "$GOML/runs/fa_v106_correct"   9

echo "================================================================"
echo "DIAGNOSTIC TABLE — what to look at:"
echo "  warps_active   : v96≈16.5%  v106≈?   (12 warps×2 blocks vs 8×2 → expect ~24%)"
echo "  wait%          : v96=37.77  v106=?   (more warps SHOULD reduce wait if latency-hiding)"
echo "  math_pipe%     : v96=8.86   v106=?   (more warps may rise math contention)"
echo "  short_scb%     : v96=6.51   v106=?"
echo "  mio_throttle%  : v96=4.65   v106=?   (SMEM port contention)"
echo "  lg_throttle%   : v96=3.31   v106=?"
echo ""
echo "EXPECTED outcomes:"
echo "  A) wait drops (12-warp helped), other stalls SAME → can't explain -15.6%"
echo "     → architectural lever lost (more blocks/grid for same work)"
echo "  B) wait drops AND math_pipe/mio rise (same as v102 bottleneck-shift)"
echo "     → confirms 12-warp on hd=128 hits MMA-pipe / SMEM-port ceiling"
echo "  C) wait UNCHANGED → original hypothesis was wrong; warps are NOT the lever"
echo "================================================================"
