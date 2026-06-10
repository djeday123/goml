#!/bin/bash
# v116 NCu vs v111: did write-conflict fix actually drop uncoalesced,
# what rose to absorb the -0.8% mean perf loss?

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

profile_uncoal() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_uncoalesced.csv"
    echo "=== UNCOAL $label cfg=$cfg ==="
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --section SourceCounters \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
    python3 -c "
import re
with open('$out') as f: lines = f.readlines()
for l in lines:
    if 'UncoalescedSharedAccess' in l:
        m = re.search(r'total of (\d[\d,]*) excessive wavefronts.*total (\d[\d,]*) wavefronts', l)
        if m:
            e, t = int(m.group(1).replace(',','')), int(m.group(2).replace(',',''))
            print(f'  excessive {e:,} of {t:,} = {100*e/t:.1f}%')
"
}

profile_stalls() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_v116_stallpct.csv"
    echo "=== STALLS $label cfg=$cfg ==="
    METRICS="smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
    python3 -c "
import csv, io
with open('$out') as f:
    lines = f.readlines()
hi = next(i for i,l in enumerate(lines) if l.startswith('\"ID\",\"Process ID\"'))
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
m = []
for r in rdr:
    n = r.get('Metric Name','')
    if 'stalled' in n and 'per_warp_active' in n:
        try: v = float(r.get('Metric Value','0').replace(',','.'))
        except: v = 0
        short = n.replace('smsp__warp_issue_stalled_','').replace('_per_warp_active.pct','')
        m.append((short, v))
m.sort(key=lambda x: -x[1])
total = sum(v for _,v in m)
for n,v in m: print(f'  {n:<24} {v:>6.2f}%')
print(f'  {\"TOTAL\":<24} {total:>6.2f}%  Eligible {100-total:.2f}%')
"
}

profile_uncoal "v111" "$GOML/runs/fa_v111_producer_skip"   9
profile_uncoal "v116" "$GOML/runs/fa_v116_swzwordrot"      9
echo ""
profile_stalls "v111" "$GOML/runs/fa_v111_producer_skip"   9
echo ""
profile_stalls "v116" "$GOML/runs/fa_v116_swzwordrot"      9
