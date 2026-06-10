#!/bin/bash
# v117 vs v111 NCu — did partial top sync actually reduce barrier %?

set -uo pipefail
NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

profile_stalls() {
    local label="$1" bin="$2"
    local out="$GOML/runs/audit_v117_${label}_stalls.csv"
    METRICS="smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct"
    "$NCU" --target-processes all --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" --csv "$bin" --ncu 9 > "$out" 2>&1
    echo "=== $label cfg=9 PEAK stalls ==="
    python3 -c "
import csv, io
with open('$out') as f: lines=f.readlines()
hi=next(i for i,l in enumerate(lines) if l.startswith('\"ID\",\"Process ID\"'))
rdr=csv.DictReader(io.StringIO(''.join(lines[hi:])))
m=[]
for r in rdr:
    n=r.get('Metric Name','')
    if 'stalled' in n and 'per_warp_active' in n:
        try: v=float(r.get('Metric Value','0').replace(',','.'))
        except: v=0
        short=n.replace('smsp__warp_issue_stalled_','').replace('_per_warp_active.pct','')
        m.append((short,v))
m.sort(key=lambda x:-x[1])
total=sum(v for _,v in m)
for n,v in m: print(f'  {n:<24} {v:>6.2f}%')
print(f'  {\"TOTAL\":<24} {total:>6.2f}%  Eligible {100-total:.2f}%')
"
}

profile_stalls "v111" runs/fa_v111_producer_skip
profile_stalls "v117" runs/fa_v117_partial_top
