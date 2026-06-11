#!/bin/bash
# v96b_pad8k: same kernel as v96b, but +8KB SMEM at launch → forces 1 block/SM.
# Critical experiment for v120 ping-pong gating: measure tensor pipe util at 1 block/SM.
# Prediction grid:
#   util ≈ 28% → second-block decorrelation was producing the 56.1% (additive model holds, v120 closed)
#   util ≥ 45% → single block already drives high util (v120 has room to fill via design)

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NCU=$CUDA/bin/ncu

cd "$GOML"

SRC=libs/flash_attention_v96b_pad8k_hd128_fp8_forward.cu
BIN=runs/fa_v96b_pad8k

echo "=== Build v96b_pad8k ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo "$SRC" -o "$BIN" -lcudart 2>&1 \
    | grep -E "stack|register|barriers"

echo ""
echo "=== Correctness + perf (cfg=9 PEAK only) ==="
"$BIN" --ncu 9

echo ""
echo "=== NCu cfg=9 full set ==="
REP="$GOML/runs/ncu_v96b_pad8k.ncu-rep"
"$NCU" --target-processes all --launch-skip 1 --launch-count 1 \
    --set full --import-source on --export "$REP" \
    "$BIN" --ncu 9 2>&1 | tail -10

echo ""
echo "=== Extract key metrics from report ==="
"$NCU" --import "$REP" --page details 2>&1 | grep -iE "tensor|pipe|smsp|achieved|active blocks|warps active|tflops" | head -20
