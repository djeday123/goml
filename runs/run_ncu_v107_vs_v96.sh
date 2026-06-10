#!/bin/bash
# NCu compare v96 vs v107 (persistent) on PEAK (cfg=9) AND wave-tail (cfg=6).
# cfg=6 = bh=16 sl=4096: total_tiles=16×32=512, 512/376=1.36 waves → wave-tail loss
#         expected on v96, v107 should equalize.
# cfg=9 = bh=64 sl=8192: total_tiles=64×64=4096, 4096/376=10.89 waves → already wave-saturated
#         v107 should = v96 (just slight outer-loop overhead).

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
EOF
)

profile() {
    local label="$1" bin="$2" cfg="$3" cfg_desc="$4"
    local out="$GOML/runs/ncu_${label}_cfg${cfg}.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg  ($cfg_desc)"
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
    print(f"  {k:<24} {v:>7.2f}%")
    total += v
print(f"  {'TOTAL':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

echo "######################################"
echo "###   PEAK (cfg=9 = bh=64 sl=8192)  ###"
echo "######################################"
profile "v96"  "$GOML/runs/fa_v96_ksbatched"  9 "PEAK bh=64 sl=8192"
profile "v107" "$GOML/runs/fa_v107_persistent" 9 "PEAK bh=64 sl=8192"

echo ""
echo "###########################################"
echo "###  WAVE-TAIL (cfg=6 = bh=16 sl=4096) ###"
echo "###########################################"
profile "v96"  "$GOML/runs/fa_v96_ksbatched"  6 "bh=16 sl=4096 total=512 1.36 waves"
profile "v107" "$GOML/runs/fa_v107_persistent" 6 "bh=16 sl=4096 total=512 → v107 grid=376 1.36 tiles/blk"

echo ""
echo "================================================================"
echo "INTERPRETATION GUIDE:"
echo "  PEAK: wave-saturated. v107 has same active warps but +outer loop overhead."
echo "        Expect slight regression in Eligible (extra sync each tile) or zero."
echo "  WAVE-TAIL: v96 launches 512 blocks (last wave at 36% efficiency)."
echo "             v107 launches 376 blocks each doing 1.36 tiles."
echo "             cycles_active should be HIGHER for v107 (longer per-block work)."
echo "             warps_active should match (same launch_bounds)."
echo "             If Eligible % similar → tile-amortization gain visible only in"
echo "             total kernel duration, not in per-cycle stall mix."
echo "================================================================"
