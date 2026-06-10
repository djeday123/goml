#!/bin/bash
# NCu compare v96 / v102 / v104 / v105 stalls + occupancy on PEAK (cfg=9).

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
    local out="$GOML/runs/ncu_${label}_stalls.csv"
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
profile "v104" "$GOML/runs/fa_v104_br96"      9
profile "v105" "$GOML/runs/fa_v105_pinregs"   9

echo "================================================================"
echo "EXPECTED v105 vs v104:"
echo "  mio_throttle ↓ (smP STS+LDS killed)"
echo "  barrier ↓ (2 syncs killed)"
echo "  short_scb possibly down (no smP smP loads)"
echo "  Eligible ↑"
echo "================================================================"
