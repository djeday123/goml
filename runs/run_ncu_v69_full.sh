#!/bin/bash
# Phase 1b: full NCu profile with PmSampling stall breakdown.
# Rebuilds v69 with -lineinfo first so source attribution works,
# then runs --set full on launch #350 (mid-bench bh=16 sl=4096).
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NCU="$CUDA/bin/ncu"
SRC="$GOML/libs/flash_attention_v69_fp8_forward.cu"
BIN_LI="$GOML/runs/fa_v69_fp8_li"
OUT="$GOML/runs/ncu_v69_full.log"
RPT="$GOML/runs/ncu_v69_full.ncu-rep"

cd "$GOML"

echo "=== Rebuilding v69 with -lineinfo (separate binary, does not touch fa_v69_fp8) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -lineinfo \
    "$SRC" \
    -o "$BIN_LI" -lcudart -Xptxas=-v 2>&1 | tail -8

echo ""
echo "=== Running NCu --set full on launch 350 (bh=16 sl=4096, mid-timing) ==="
"$NCU" \
    --target-processes all \
    --launch-skip 350 --launch-count 1 \
    --set full \
    --import-source yes \
    --export "$RPT" \
    --csv --page details \
    "$BIN_LI" > "$OUT" 2>&1 || true

echo ""
echo "=== Extracting key sections from $OUT ==="
echo ""
echo "--- Warp State Statistics (stall breakdown) ---"
grep -E "Warp State Statistics|Stall|Active Warps|No Eligible|Eligible Warps|Warp Cycles" "$OUT" | head -40

echo ""
echo "--- Source counters (top stalls) ---"
grep -E "Source Counters|Sample|Sampling" "$OUT" | head -20

echo ""
echo "--- Pipeline utilization ---"
grep -E "Pipe Util|SOL|Tensor|FMA|Pipe Throughput" "$OUT" | head -30

echo ""
echo "Report: $RPT (open with: $NCU --import $RPT --page details)"
