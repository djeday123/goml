#!/bin/bash
# NCu profile v83 TMA-K on win + loss configs vs v81 baseline (already measured).
# Win config:  bh=16 sl=4096 (idx 6) — grid 512, WIN regime in v81 LB=3
# Loss config: bh=8 sl=4096 (idx 4) — grid 256, was LOSS in v81 LB=3, became WIN in v83
# Diagnostic: did No Eligible drop in WIN/LOSS LB=3 vs v81 (43% / 59.56%)?
set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v83_tmak_fp8"

cd "$GOML"

if [ ! -x "$NCU" ]; then echo "ERROR: ncu not at $NCU" >&2; exit 1; fi
if [ ! -x "$BIN" ]; then echo "ERROR: binary not at $BIN — run run_fa_v83_tmak.sh first" >&2; exit 1; fi

echo "=== Standalone sanity (no profiler) ==="
"$BIN" --ncu 6 3 2>&1 | head -5
echo ""

profile_one() {
    local label="$1" cfg="$2" lb="$3"
    local logf="$GOML/runs/ncu_v83_cfg${cfg}_lb${lb}.log"
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
        tail -30 "$logf"
        return $rc
    fi
    echo "--- key metrics ---"
    grep -E "(Achieved Occupancy|Theoretical Occupancy|Active Warps|Eligible Warps|Issue Slot Util|Waves Per|Block Limit|No Eligible|Stall|Compute \(SM\)|Memory \[%\]|Shared.*Bank)" "$logf" | head -32
    echo ""
}

profile_one "WIN  LB=2 (TMA-K, 2 blocks/SM)"  6 2
profile_one "WIN  LB=3 (TMA-K, 3 blocks/SM)"  6 3
profile_one "LOSS LB=2 (TMA-K, 2 blocks/SM)"  4 2
profile_one "LOSS LB=3 (TMA-K, 3 blocks/SM)"  4 3

echo ""
echo "================================================================"
echo "Compare with v81 baseline (from sm120-fa-v81-ncu-mechanism memory):"
echo "  WIN  LB=3 v81: Achieved Occ 21.15%, No Eligible 43.28%, Compute 51.73%, Waves 0.91"
echo "  LOSS LB=3 v81: Achieved Occ 11.62%, No Eligible 59.56%, Compute 34.06%, Waves 0.45"
echo "If v83 WIN LB=3: No Eligible <43% AND Compute >52% AND perf -1.3% → TMA WORKS, overhead eats it"
echo "If v83 WIN LB=3: No Eligible >=43% OR Compute ~52% (same) → TMA doesn't help latency"
echo "================================================================"
