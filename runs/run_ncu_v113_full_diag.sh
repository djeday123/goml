#!/bin/bash
# Full diagnostic: v113 stalls + uncoalesced for v96/v111/v113 (proper CSV parse).

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

# Standard stall metrics
METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
launch__block_size
launch__registers_per_thread
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
smsp__warp_issue_stalled_drain_per_warp_active.pct
smsp__warp_issue_stalled_not_selected_per_warp_active.pct
smsp__warp_issue_stalled_selected_per_warp_active.pct
EOF
)

profile_stalls() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_stalls.csv"
    echo "================================================================"
    echo "  STALL BREAKDOWN: $label  cfg=$cfg"
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
print("  warps_active:", launch_metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active','?'))
print("  cycles_active:", launch_metrics.get('sm__cycles_active.avg.pct_of_peak_sustained_elapsed','?'))
print("  STALLS:")
real_total = 0
for k,v in sorted(stall_metrics.items(), key=lambda x:-x[1]):
    if v > 0.05:
        print(f"    {k:<24} {v:>7.2f}%")
        if k not in ('selected','not_selected'):
            real_total += v
print(f"    {'REAL STALLS':<24} {real_total:>7.2f}%")
print(f"    {'Real Eligible':<24} {100-real_total:>7.2f}%")
PYEOF
    echo ""
}

# Profile uncoalesced from SourceCounters
profile_uncoalesced() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_src.csv"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --section SourceCounters \
        --section InstructionStats \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
}

extract_uncoalesced() {
    local label="$1"
    local out="$GOML/runs/ncu_${label}_src.csv"
    python3 << PYEOF
import csv, io, re
with open("$out") as f: lines = f.readlines()
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'):
        hi = i; break
if hi is None: raise SystemExit(0)
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
print(f"  --- {'$label':<6} ---")
for r in rdr:
    rule = r.get('Rule Name','')
    if 'Uncoalesced' in rule:
        desc = r.get('Rule Description','')
        speedup = r.get('Estimated Speedup','')
        # Extract excessive wavefronts
        m = re.search(r'total of ([\d,]+) excessive (\w+) of the total ([\d,]+) (\w+)', desc)
        if m:
            exc_num = int(m.group(1).replace(',',''))
            tot_num = int(m.group(3).replace(',',''))
            unit = m.group(2)
            pct = 100.0 * exc_num / tot_num if tot_num > 0 else 0
            print(f"    {rule:<28} excessive {exc_num:>12,} / {tot_num:>12,} {unit} = {pct:5.1f}%   speedup hint={speedup}%")
        else:
            print(f"    {rule:<28} (no parse) speedup hint={speedup}%")
PYEOF
}

echo "######################################"
echo "###  STALL BREAKDOWN (PEAK cfg=9)  ###"
echo "######################################"
profile_stalls "v110" "$GOML/runs/fa_v110_warpspec_1p3c" 9
profile_stalls "v111" "$GOML/runs/fa_v111_producer_skip" 9
profile_stalls "v113" "$GOML/runs/fa_v113_producer_arrive" 9

echo "######################################"
echo "###  UNCOALESCED SHARED ACCESS    ###"
echo "######################################"
profile_uncoalesced "v96"  "$GOML/runs/fa_v96_ksbatched"
profile_uncoalesced "v111" "$GOML/runs/fa_v111_producer_skip"
profile_uncoalesced "v113" "$GOML/runs/fa_v113_producer_arrive"
extract_uncoalesced "v96"
extract_uncoalesced "v111"
extract_uncoalesced "v113"

echo ""
echo "================================================================"
echo "INTERPRETATION:"
echo ""
echo "STALLS — focus on:"
echo "  long_scoreboard: producer cp.async waiting (memory-bound producer)"
echo "  short_scoreboard: consumer SMEM/math (math-bound consumer)"
echo "  → if long_scb dominant → ldmatrix/deep buffer helps"
echo "  → if short_scb dominant → ldmatrix doesn't help (consumer-math bound)"
echo ""
echo "UNCOALESCED — focus on:"
echo "  If v96 SharedAccess is much lower than v111/v113 → our consumer-stride"
echo "    transpose introduces conflicts (fixable!)"
echo "  If similar → baseline issue"
echo "================================================================"
