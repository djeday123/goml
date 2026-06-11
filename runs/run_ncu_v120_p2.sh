#!/bin/bash
# P2: NCu v120a/b на cfg=9 — для сравнения с v96b baseline.
set -uo pipefail
NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

UTIL_METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
EOF
)
STALL_METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
EOF
)

profile() {
    local bin="$1" tag="$2"
    local out=runs/ncu_${tag}_cfg9.csv
    "$NCU" --target-processes all --launch-skip 1 --launch-count 1 \
        --metrics "$UTIL_METRICS,$STALL_METRICS" --csv \
        "$bin" --ncu 9 > "$out" 2>&1
    python3 - "$out" "$tag" << 'PYEOF'
import csv, io, sys
out, tag = sys.argv[1], sys.argv[2]
with open(out) as f: lines = f.readlines()
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'): hi = i; break
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
metrics = {}
for r in rdr:
    n = r.get('Metric Name','')
    try: v = float(r.get('Metric Value','0').replace(',','.'))
    except: v = 0
    if n: metrics[n] = v
print(f"--- {tag} ---")
print(f"  tensor util (active) : {metrics.get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active', 0):.2f}%")
print(f"  sm throughput        : {metrics.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0):.2f}%")
print(f"  warps_active         : {metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0):.2f}%")
print(f"  wait                 : {metrics.get('smsp__warp_issue_stalled_wait_per_warp_active.pct', 0):.2f}%")
print(f"  math_pipe            : {metrics.get('smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct', 0):.2f}%")
print(f"  short_scb            : {metrics.get('smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct', 0):.2f}%")
print(f"  mio                  : {metrics.get('smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct', 0):.2f}%")
print(f"  barrier              : {metrics.get('smsp__warp_issue_stalled_barrier_per_warp_active.pct', 0):.2f}%")
print(f"  long_scb             : {metrics.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', 0):.2f}%")
PYEOF
}

profile runs/fa_v96b_localfix       v96b_p2
profile runs/fa_v120a_halfphase     v120a_p2
profile runs/fa_v120b_quarterphase  v120b_p2
