#!/bin/bash
# NCu probe for v102: did 3 blocks/SM actually load? + stall composition.
# Compare to v96 (2 blocks/SM) baseline.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN_V96="$GOML/runs/fa_v96_ksbatched"
BIN_V102="$GOML/runs/fa_v102_nosmvt"

cd "$GOML"

if [ ! -x "$NCU" ]; then echo "ncu missing"; exit 1; fi
if [ ! -x "$BIN_V96" ]; then echo "v96 missing"; exit 1; fi
if [ ! -x "$BIN_V102" ]; then echo "v102 missing"; exit 1; fi

# Need v102 to have --ncu mode too. Let me just check if it has it via the binary
# args — if not, we'll use --launch-skip on full bench.

METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
launch__block_size
launch__grid_size
launch__registers_per_thread
launch__shared_mem_per_block_dynamic
launch__shared_mem_per_block_static
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
launch__occupancy_per_block_size
launch__occupancy_per_register_size
launch__occupancy_per_shared_mem_size
sm__warps_active.avg.pct_of_peak_sustained_active
sm__warps_active.avg.peak_sustained_active
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
    local out="$GOML/runs/ncu_${label}_occupancy.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
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
    print("  ERROR: header not found"); raise SystemExit(0)
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
rows = list(rdr)
launch_metrics = {}
stall_metrics = {}
for r in rows:
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
for k in ['launch__block_size', 'launch__grid_size', 'launch__registers_per_thread',
         'launch__shared_mem_per_block_dynamic', 'launch__shared_mem_per_block_static',
         'launch__occupancy_limit_blocks', 'launch__occupancy_limit_registers',
         'launch__occupancy_limit_shared_mem', 'launch__occupancy_limit_warps',
         'launch__occupancy_per_block_size', 'launch__occupancy_per_register_size',
         'launch__occupancy_per_shared_mem_size',
         'sm__warps_active.avg.pct_of_peak_sustained_active',
         'sm__warps_active.avg.peak_sustained_active']:
    if k in launch_metrics:
        print(f"  {k.replace('launch__',''):<48} {launch_metrics[k]:>10.2f}")
print()
print("--- STALLS ---")
total = 0
for k, v in sorted(stall_metrics.items(), key=lambda x: -x[1]):
    print(f"  {k:<24} {v:>7.2f}%")
    total += v
print(f"  {'TOTAL stalled':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

profile "v96"  "$BIN_V96"  9
profile "v102" "$BIN_V102" 9

echo "================================================================"
echo "INTERPRETATION:"
echo "  occupancy_limit_blocks shows the smaller of all per-block limits."
echo "  If v102 occupancy_limit > v96 → 3-block budget fits → check actual"
echo "    sm__warps_active.peak_sustained_active to see if it actually went up."
echo "  Compare warps active: v96 has ~8/SM at 2 blocks; v102 should be 12 if 3 loaded."
echo "  If v102 active warps ≈ v96 → 3 blocks didn't load OR scheduler limit kicked in."
echo "================================================================"
