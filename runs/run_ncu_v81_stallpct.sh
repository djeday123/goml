#!/bin/bash
# Re-query NCu with EXPLICIT stall-% metrics (per warp active) to get distributed
# fractional percentages, not the suspicious "1.0/0" pattern from pmsampling .avg.
set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v81_hd64_fp8"

cd "$GOML"

# Standard NCu metric names for stall composition (% per warp active).
# These DO sum to roughly the No Eligible % (43% in v81 WIN LB=3).
METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
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

profile_cfg() {
    local label="$1" cfg="$2" lb="$3"
    local logf="$GOML/runs/ncu_v81_stallpct_cfg${cfg}_lb${lb}.csv"

    echo "================================================================"
    echo "  $label  cfg=$cfg LB=$lb"
    echo "================================================================"

    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$BIN" --ncu "$cfg" "$lb" > "$logf" 2>&1

    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "ncu exit $rc — log tail:"
        tail -20 "$logf"
        return $rc
    fi

    echo "--- Stall % composition ---"
    # CSV format: kernel metadata cols, then metric columns.
    # Extract the stall metric columns + their values.
    python3 << PYEOF
import csv, sys
with open("$logf") as f:
    rdr = csv.reader(f)
    rows = list(rdr)
if len(rows) < 2:
    print("Empty CSV")
    sys.exit(0)
header = rows[0]
# NCu CSV puts units in row 2, values from row 3+
if len(rows) >= 3:
    units = rows[1]
    data = rows[2]
else:
    units = [''] * len(header)
    data = rows[1]
stalls = []
for i, col in enumerate(header):
    if 'stalled' in col.lower() and 'per_warp_active' in col.lower():
        try:
            val = float(data[i]) if data[i] not in ('', '""', 'n/a') else 0.0
        except (ValueError, IndexError):
            val = 0.0
        short = col.replace('smsp__warp_issue_stalled_', '').replace('_per_warp_active.pct', '').strip('"')
        stalls.append((short, val))
stalls.sort(key=lambda x: x[1], reverse=True)
total = sum(v for _, v in stalls)
print(f"  {'Stall reason':<24} {'%':>8}")
print('  ' + '-' * 34)
for name, val in stalls:
    print(f"  {name:<24} {val:>7.2f}%")
print('  ' + '-' * 34)
print(f"  {'TOTAL (≈ No Eligible %)':<24} {total:>7.2f}%")
PYEOF
    echo ""
}

profile_cfg "WIN  LB=3 (perf 390T, +27%)" 6 3
profile_cfg "WIN  LB=2 (perf 308T, baseline)" 6 2
profile_cfg "LOSS LB=3 (perf 261T, -8%)"  4 3
profile_cfg "LOSS LB=2 (perf 284T)"  4 2
