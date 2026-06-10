#!/bin/bash
# v115 SHFL transpose NCu profile vs v96.
# Hypothesis: UncoalescedSharedAccess drops (4-way write conflict gone),
# but math_pipe_throttle rises (SHFL+byte-permutation pumped into MMA pipe).

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

profile_uncoal() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_uncoalesced.csv"
    echo "================================================================"
    echo "  UNCOAL $label  cfg=$cfg"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --section SourceCounters \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
    python3 << PYEOF
import re
with open("$out") as f: lines = f.readlines()
for l in lines:
    if 'UncoalescedSharedAccess' in l:
        parts = l.split(',')
        if len(parts) > 17:
            desc = parts[17] if len(parts) > 17 else ''
            speedup = parts[19].strip().strip('"') if len(parts) > 19 else ''
            print(f"  UncoalescedShared    speedup hint: {speedup}")
            m = re.search(r'total of (\d[\d,]*) excessive wavefronts.*total (\d[\d,]*) wavefronts', desc)
            if m:
                exc, tot = m.groups()
                e, t = int(exc.replace(',','')), int(tot.replace(',',''))
                print(f"    excessive: {e:,}  of total {t:,}  ({100*e/t:.1f}%)")
PYEOF
    echo ""
}

profile_stalls() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_stalls.csv"
    echo "================================================================"
    echo "  STALLS $label  cfg=$cfg"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --section WarpStateStats \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
    python3 << PYEOF
import csv
with open("$out") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 13: continue
        name = row[12] if len(row) > 12 else ''
        if 'smsp__pcsamp_warps_issue_stalled' in name and 'per_warp_active' in name:
            short = name.replace('smsp__pcsamp_warps_issue_stalled_','').replace('.pct_per_warp_active','')
            val = row[14] if len(row) > 14 else ''
            print(f"  {short[:30]:<30} {val}")
        elif name.startswith('smsp__average_warps_issue_stalled'):
            short = name.replace('smsp__average_warps_issue_stalled_','').replace('.pct','')
            val = row[14] if len(row) > 14 else ''
            print(f"  AVG {short[:26]:<26} {val}")
PYEOF
    echo ""
}

# Config 9 = bh=64 sl=8192 = PEAK config
profile_uncoal "v96"  "$GOML/runs/fa_v96_ksbatched"            9
profile_uncoal "v115" "$GOML/runs/fa_v115_shfl_transpose"      9

profile_stalls "v96"  "$GOML/runs/fa_v96_ksbatched"            9
profile_stalls "v115" "$GOML/runs/fa_v115_shfl_transpose"      9

echo "================================================================"
echo "  Predictions:"
echo "  v115 UncoalescedShared: DROP from 40% (write conflict eliminated)"
echo "  v115 wait stall:        similar or DROP"
echo "  v115 math_pipe_throttle: RISE (SHFL+byte perm in MMA pipe)"
echo "  v115 short_scoreboard:  RISE (LDS.32 instead of LDS.U8, MIO contention)"
echo "================================================================"
