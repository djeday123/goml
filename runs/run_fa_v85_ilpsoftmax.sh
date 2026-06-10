#!/bin/bash
# v85 hd=64 + ILP softmax — Phase E split (ex2 issue loop separate from ns accum).
# Target: wait stall 28.40% → 15-20% in WIN LB=3 → +5-10% perf.
# Zero register cost (just reorder existing data flow). Compare ptxas to v81's 168 regs LB=3.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v85 ILP softmax (LB=2 and LB=3, -lineinfo for future NCu) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v85_hd64_ilpsoftmax_fp8_forward.cu \
    -o runs/fa_v85_ilpsoftmax_fp8 -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Run: attrs + correctness ×8×2 + A/B perf ×11×2 ×variance 3 ==="
runs/fa_v85_ilpsoftmax_fp8
