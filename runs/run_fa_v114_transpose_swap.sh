#!/bin/bash
# v114 = v96 + transpose_v thread mapping SWAP.
# Original mapping: tk=t/tiles_n, tn=t%tiles_n (adjacent threads = adjacent N-rows of smV_T)
# v114 mapping:     tk=t%tiles_k, tn=t/tiles_k (adjacent threads = adjacent K-cols)
# Goal: reduce 4-way smV_T WRITE bank conflict → 2-way.
# Risk: smV READ pattern may worsen.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v114 transpose SWAP ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v114_transpose_swap_hd128_fp8_forward.cu \
    -o runs/fa_v114_transpose_swap -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 242 regs, 568 mean PEAK, UncoalescedSharedAccess 40.0%"
echo "  v114 target: same regs, hopefully UncoalescedSharedAccess <30% AND perf ≥ v96"
echo ""
echo "=== Run: full correctness + bench ==="
runs/fa_v114_transpose_swap
