#!/bin/bash
# Full v116 audit: build + correctness + perf A/B vs v111 + NCu (UncoalescedShared + stalls).
# Reproduces the full result chain described in the v116 closure.

set -uo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NCU=$CUDA/bin/ncu

cd "$GOML"

V111_BIN=runs/fa_v111_producer_skip
V116_BIN=runs/fa_v116_swzwordrot
V111_SRC=libs/flash_attention_v111_producer_skip_hd128_fp8_forward.cu
V116_SRC=libs/flash_attention_v116_swzwordrot_hd128_fp8_forward.cu

echo "############################################################"
echo "# Step 1: Build v116 (and rebuild v111 if missing)"
echo "############################################################"

if [ ! -x "$V111_BIN" ]; then
    echo "--- Build v111 baseline ---"
    "$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
        -Xptxas=-v -lineinfo "$V111_SRC" -o "$V111_BIN" -lcudart 2>&1 \
        | grep -E "(register|spill|error)" | head -5
fi

echo "--- Build v116 ---"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo "$V116_SRC" -o "$V116_BIN" -lcudart 2>&1 \
    | grep -E "(register|spill|stack|error)" | head -5

echo ""
echo "############################################################"
echo "# Step 2: v111 baseline perf (correctness + bench)"
echo "############################################################"
"$V111_BIN" | sed -n '/Correctness/,$p'

echo ""
echo "############################################################"
echo "# Step 3: v116 perf (correctness + bench, same thermal state)"
echo "############################################################"
"$V116_BIN" | sed -n '/Correctness/,$p'

echo ""
echo "############################################################"
echo "# Step 4: NCu — UncoalescedSharedAccess (target metric)"
echo "############################################################"

for label_bin in "v111:$V111_BIN" "v116:$V116_BIN"; do
    label="${label_bin%%:*}"
    bin="${label_bin##*:}"
    out="$GOML/runs/audit_${label}_uncoal.csv"
    echo "--- $label cfg=9 (bh=64 sl=8192 PEAK) ---"
    "$NCU" --target-processes all --launch-skip 1 --launch-count 1 \
        --section SourceCounters --csv "$bin" --ncu 9 > "$out" 2>&1
    python3 -c "
import re
with open('$out') as f: lines = f.readlines()
for l in lines:
    if 'UncoalescedSharedAccess' in l:
        m = re.search(r'total of (\d[\d,]*) excessive wavefronts.*total (\d[\d,]*) wavefronts', l)
        if m:
            e = int(m.group(1).replace(',',''))
            t = int(m.group(2).replace(',',''))
            print(f'  UncoalescedShared: {100*e/t:.1f}%  (excessive {e:,} of {t:,})')
        break
"
done

echo ""
echo "############################################################"
echo "# Step 5: NCu — Stall composition (cfg=9 PEAK)"
echo "############################################################"

METRICS=$(echo "
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct
" | tr '\n' ',' | sed 's/^,//;s/,$//')

for label_bin in "v111:$V111_BIN" "v116:$V116_BIN"; do
    label="${label_bin%%:*}"
    bin="${label_bin##*:}"
    out="$GOML/runs/audit_${label}_stalls.csv"
    echo "--- $label stalls ---"
    "$NCU" --target-processes all --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" --csv "$bin" --ncu 9 > "$out" 2>&1
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
print(f'  {\"TOTAL stalled\":<24} {total:>6.2f}%')
print(f'  {\"Eligible\":<24} {100-total:>6.2f}%')
"
done

echo ""
echo "############################################################"
echo "# Summary"
echo "############################################################"
echo "Expected v111 vs v116 PEAK (bh=64 sl=8192):"
echo "  v111: 489.1 T / Eligible 28.1% / UncoalescedShared 40.2%"
echo "  v116: 481.6 T / Eligible 28.1% / UncoalescedShared 30.0%"
echo ""
echo "Mechanism: UncoalescedShared dropped −10.2pp as math predicted,"
echo "but Eligible % unchanged (architectural fixed-point) → wall-clock −1.5%"
echo "due to extra ALU ops in swz_byte_smvt (+3 regs)."
echo ""
echo "Conclusion: 12pp transpose_v uncoalesced is a structural floor empirically."
