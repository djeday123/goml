#!/bin/bash
# NCu profile v81 hd=64 on two configs × two LBs.
# Mirrors run_ncu_v69_phase1.sh working pattern.
#
# Win config:  bh=16 sl=4096 (idx 6) — grid 512, LB=2 needs 2 waves, LB=3 fits 1
# Loss config: bh=8  sl=4096 (idx 4) — grid 256, both fit < 1 wave
set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v81_hd64_fp8"

cd "$GOML"

if [ ! -x "$NCU" ]; then
    echo "ERROR: ncu not at $NCU" >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "ERROR: binary not built at $BIN — run run_fa_v81_hd64.sh first" >&2
    exit 1
fi

# Confirm --ncu mode works standalone (no profiler) on win config.
echo "=== Standalone sanity (no profiler) ==="
"$BIN" --ncu 6 3 2>&1 | head -5
echo ""

profile_one() {
    local label="$1" cfg="$2" lb="$3"
    local logf="$GOML/runs/ncu_v81_cfg${cfg}_lb${lb}.log"
    echo "================================================================"
    echo "  $label   cfg_idx=$cfg LB=$lb  → $logf"
    echo "================================================================"

    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --section WarpStateStats \
        --section SchedulerStats \
        --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis \
        --section SpeedOfLight \
        --section Occupancy \
        --section LaunchStats \
        --csv --page details \
        "$BIN" --ncu "$cfg" "$lb" > "$logf" 2>&1
    local rc=$?
    echo "ncu exit: $rc  ($(wc -l < "$logf") lines in log)"

    if [ $rc -ne 0 ] || [ $(wc -l < "$logf") -lt 5 ]; then
        echo "--- log tail (showing error) ---"
        tail -30 "$logf"
        return $rc
    fi

    # Print key metrics from CSV.
    echo "--- key metrics ---"
    grep -E "(Achieved Occupancy|Theoretical Occupancy|Active Warps|Eligible Warps|Issue Slot Util|Waves Per|Block Limit|No Eligible|Stall|Compute \(SM\)|Memory \[%\])" "$logf" | head -30
    echo ""
}

profile_one "WIN  LB=2 baseline (2 blocks/SM)"  6 2
profile_one "WIN  LB=3            (3 blocks/SM)" 6 3
profile_one "LOSS LB=2 baseline (2 blocks/SM)"  4 2
profile_one "LOSS LB=3            (3 blocks/SM)" 4 3

echo ""
echo "================================================================"
echo "Mechanism check:"
echo "  WIN  LB=3 expected: Achieved Occ ~12.5→18.75%, No Eligible ↓, Waves ↓"
echo "  LOSS LB=3 expected: Achieved Occ ↑ too, but No Eligible same/↑, Waves same"
echo "================================================================"
