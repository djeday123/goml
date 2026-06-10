#!/bin/bash
# v101 = v96 + single-K (drop smK[1] double-buffer).
# Step 1 of 3-block path. SMEM: 48.5KB → 40.5KB per block (saves 8KB).
#
# Changes vs v96:
#   - SMEM layout: smK[2] → smK (single)
#   - V always lands in smV (no alternating prev_V_slot via smK[buf])
#   - K[kv+1] cp.async DEFERRED to after QK done (was at iter start)
#     → less overlap (only softmax+PV vs full iter in v96)
#   - V[kv+1] cp.async target: smK[buf] → smV
#
# RISK ANALYSIS:
#   K cp.async overlap shrinks ~50% → long_scoreboard likely grows
#   Expected perf delta: -2 to -5% on peak (acceptable cost for SMEM saving)
#   If correctness 8/8 PASS → proceed to v102 (smV_T elim via ldmatrix.trans)
#
# 3 BLOCKS check: 40.5 × 3 = 121.5 KB > 100 KB cap. Still NOT enough.
# Need v102 to drop smV_T (8.7 KB) → 31.8 × 3 = 95 KB ≤ 100 KB ✅

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v101 hd=128 single-K ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v101_singleK_hd128_fp8_forward.cu \
    -o runs/fa_v101_singleK -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference (v96 sustained) ==="
echo "  v96: 242 regs, 48.5KB/block, 568.5T (3-run) / 564.4T (30-run sustained)"
echo "  v101 prediction: regs ≈ 242 (kernel logic same, no extra arrays)"
echo "  v101 SMEM: 40.7KB (-8KB), still 2 blocks/SM (3-block needs v102)"
echo ""
echo "=== Run: correctness 8/8 + bench ==="
runs/fa_v101_singleK

echo ""
echo "=== Watch ==="
echo "  Correctness: MUST 8/8 PASS (race conditions are biggest risk)"
echo "  Peak (bh=64 sl=8192) vs v96 568.5T:"
echo "    ≈ v96 (566-570T)        → null cost from single-K, proceed to v102"
echo "    < v96 by 2-5% (540-555T) → expected cost, still OK for 3-block path"
echo "    < v96 by 10%+           → high cost; reconsider"
