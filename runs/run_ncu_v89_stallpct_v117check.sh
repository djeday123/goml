#!/bin/bash
# Quick check: is v89 hd=64 at architectural fixed-point like v96/v111 hd=128?
# If barrier % is small (<3%), barrier-reduction direction has no room on hd=64.
# If barrier % is large (>5%), direct attack might give same null-conversion pattern.

set -uo pipefail
NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v89_pinregs_fp8"
cd "$GOML"

METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct
EOF
)

profile_cfg() {
    local cfg="$1" label="$2"
    local out="$GOML/runs/ncu_v89_v117check_cfg${cfg}.csv"
    echo "=== v89 hd=64 cfg=$cfg ($label) ==="
    "$NCU" --target-processes all --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" --csv "$BIN" --ncu "$cfg" > "$out" 2>&1
    python3 -c "
import csv, io
with open('$out') as f: lines=f.readlines()
hi=next((i for i,l in enumerate(lines) if l.startswith('\"ID\",\"Process ID\"')),None)
if hi is None: print('  ERROR no header'); exit(0)
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
print(f'  TOTAL                    {total:>6.2f}%  Eligible {100-total:.2f}%')
"
}

# cfg=9 = bh=64 sl=8192 = WIN regime for v89 hd=64 (LB=3, 413T)
# cfg=6 = bh=16 sl=4096 = mid grid
profile_cfg 9 "WIN bh=64 sl=8192"
echo ""
profile_cfg 6 "MID bh=16 sl=4096"
