#!/bin/bash
# P0: NCu профиль v111-mbarrier на cfg=9 (bh=64 sl=8192 = peak).
# Снимаем: stall composition + tensor pipe util + compute SM util.
# Сравнение с v96b baseline (нужно отдельно снять/найти).

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN_V111="$GOML/runs/fa_v111_warpspec_mbarrier"

cd "$GOML"

if [ ! -x "$NCU" ]; then echo "ERROR: ncu not at $NCU" >&2; exit 1; fi
if [ ! -x "$BIN_V111" ]; then echo "ERROR: $BIN_V111 not built" >&2; exit 1; fi

# Метрики:
#   stall_pct (14 категорий)
#   tensor pipe util — главное число
#   compute/SM throughput, achieved occupancy
STALL_METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct
smsp__warp_issue_stalled_drain_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
smsp__warp_issue_stalled_misc_per_warp_active.pct
smsp__warp_issue_stalled_sleeping_per_warp_active.pct
smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct
smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct
EOF
)

UTIL_METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
EOF
)

profile() {
    local label="$1" cfg="$2" bin="$3" tag="$4"
    local outs="$GOML/runs/ncu_${tag}_stallpct_cfg${cfg}.csv"
    local outu="$GOML/runs/ncu_${tag}_util_cfg${cfg}.csv"

    echo "================================================================"
    echo "  $label  cfg=$cfg"
    echo "================================================================"

    "$NCU" --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$STALL_METRICS" --csv \
        "$bin" --ncu "$cfg" > "$outs" 2>&1
    echo "  wrote $outs"

    "$NCU" --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$UTIL_METRICS" --csv \
        "$bin" --ncu "$cfg" > "$outu" 2>&1
    echo "  wrote $outu"

    python3 - "$outs" "$outu" "$label" << 'PYEOF'
import csv, io, sys
outs, outu, label = sys.argv[1], sys.argv[2], sys.argv[3]

def parse_metrics(path):
    with open(path) as f: lines = f.readlines()
    hi = None
    for i, l in enumerate(lines):
        if l.startswith('"ID","Process ID"'): hi = i; break
    if hi is None: return []
    rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
    out = []
    for r in rdr:
        n = r.get('Metric Name','')
        try: v = float(r.get('Metric Value','0').replace(',','.'))
        except: v = 0
        if n: out.append((n, v))
    return out

print(f"\n--- {label} ---")

stalls = parse_metrics(outs)
stall_items = []
for n, v in stalls:
    if 'stalled' in n.lower() and 'per_warp_active' in n.lower():
        short = n.replace('smsp__warp_issue_stalled_','').replace('_per_warp_active.pct','')
        stall_items.append((short, v))
stall_items.sort(key=lambda x: x[1], reverse=True)
total = sum(v for _,v in stall_items)
print("  STALLS:")
for n,v in stall_items:
    print(f"    {n:<24} {v:>7.2f}%")
print(f"    {'TOTAL stalled':<24} {total:>7.2f}%")
print(f"    {'Eligible':<24} {100-total:>7.2f}%")

utils = parse_metrics(outu)
print("  UTILIZATION:")
for n, v in utils:
    short = n.replace('sm__pipe_','').replace('sm__inst_executed_pipe_','pipe_')
    print(f"    {short:<70} {v:>7.2f}%")
PYEOF
}

profile "v111-mbarrier cfg=9 (peak)" 9 "$BIN_V111" "v111_mbarrier"
