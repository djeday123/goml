#!/bin/bash
# v100 = v96 + K[kv+1] cp.async split into 2 groups (half-tile each).
#
# CHANGE: replace single load_tile_fp8(nxt_K, ..., FA_BC, ...) with
#   load_tile_fp8_range(nxt_K, ..., 0, FA_BC/2, ...)    // rows 0..31
#   cpa_commit                                            // group A
#   load_tile_fp8_range(nxt_K, ..., FA_BC/2, FA_BC, ...) // rows 32..63
#   cpa_commit                                            // group B
#
# HYPOTHESIS: maybe finer-grain groups change scheduler/HBM behavior.
# SKEPTICAL: V cp.async (committed last in iter) is the wait bottleneck per
# commit-order analysis. Split K doesn't speed up V completion.
# Most likely: NULL or slight regression from extra cpa_commit overhead.
#
# Falsifiable: NCu will show if long_scoreboard changes. If null → confirms
# V-bound. If positive → opens "split everything" lever direction.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v100 hd=128 K cp.async split ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v100_ksplit_hd128_fp8_forward.cu \
    -o runs/fa_v100_ksplit -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference (v96 sustained) ==="
echo "  v96: 242 regs, peak 568.5T (3-run) / 564.4T (30-run sustained)"
echo "  v96 long_scb peak: 1.77%"
echo "  v99 .ca was REJECTED (peak -3.2%) — L1 pollution"
echo ""
echo "=== Run: correctness + bench ==="
runs/fa_v100_ksplit

echo ""
echo "=== Watch ==="
echo "  v100 peak vs v96 568.5T (3-run baseline):"
echo "    > v96 → split K helped (mechanism exists, NCu next)"
echo "    ≈ v96 → null (V-bound confirmed)"
echo "    < v96 → split overhead hurts (commit cost > gain)"
