#!/bin/bash
# v97 = v96 + P-in-regs (port v89 hd=64 mechanism to hd=128).
#
# CONTEXT: P-in-regs ALREADY FAILED ONCE on hd=128 (v79c2, -2 to -5%). However
# v79c2 base was v79b which lacked ks-batched MMA. v89's success on hd=64 was
# specifically on v87 base (had ks-batched + 8-reg savings). v96 now provides
# the same pre-condition on hd=128 (ks-batched, -7 regs, +6.5-7.1% perf).
#
# HYPOTHESIS: "ks-batched base fixes P-in-regs" — falsifiable.
#   v97 > v96 → reorder-base unlocked P-in-regs on hd=128. New champion ~590T.
#   v97 ≈/< v96 → P-in-regs structurally fails on hd=128 regardless of base.
#                  Closed after 2 attempts (v79c2 + v97). v96 = final champion.
#
# CODE CHANGES:
#   1. Skip __syncthreads + smP STS loop + __syncthreads. Pack into Pf_pair[8][2].
#   2. PV ks=0 reads Pf_pair[0..3] via shfl_sync (uniform-nt pattern).
#   3. PV ks=1 reads Pf_pair[4..7]. Same pattern.
#   4. nt loop in PV stays 16 (hd=128) — only the gather pre-loop changed.
#
# REG BUDGET WATCH:
#   v96: 242 regs (2 blocks/SM ceiling = 256). Headroom = 14 regs.
#   v89 hd=64 added 8 regs for shfl machinery (160→168).
#   v97 prediction: 242→~250 regs. If 257+ → drops to 1 block/SM → catastrophe.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v97 hd=128 P-in-regs ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v97_pinregs_hd128_fp8_forward.cu \
    -o runs/fa_v97_pinregs -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "  v79b: 249 regs, 401T (small) / 534T (large peak at bh=64 sl=8192)"
echo "  v96:  242 regs (-7), 568T (large peak, +6.5%)"
echo "  v89 hd=64 analog: +8 regs (160→168), +3.8% wide"
echo "  v79c2 (P-in-regs on v79b BASE, NOT v96): FAILED -2 to -5%"
echo ""
echo "=== Watch: regs >256 → drop to 1 block/SM = catastrophe ==="
echo ""
echo "=== Run: correctness + bench (small + large + sliding window) ==="
runs/fa_v97_pinregs
