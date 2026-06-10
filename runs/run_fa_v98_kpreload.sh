#!/bin/bash
# v98 = v96 + K-preload (port v90 mechanism to hd=128).
#
# HYPOTHESIS: v96 NCu showed short_scoreboard 6.50% at peak (bh=64 sl=8192).
# K-preload separates LDS from MMA consumer — scheduler can issue MMA on
# already-prefetched K bytes without waiting on SMEM dependency chain.
#
# v90 hd=64 was null (regs 168→168, perf 0%) — compiler already optimal there.
# hd=128 different regime: 4 QK ks-batches (vs 2), 2× SMEM reads per kv-iter,
# wait 37% vs hd=64's 28%. Possibly responds where hd=64 didn't.
#
# CHANGE: in each of 4 QK ks-batches, declare b0_arr[8]/b1_arr[8], preload
# 16 K bytes from cur_K BEFORE nt MMA loop. MMA reads from arrays.
# PV V loads unchanged (not preloaded — compound risk per v91 lesson).
#
# REG BUDGET: v96 = 242. K-preload adds ~16 regs live during MMA loop per batch.
# Compiler may already hoist (v90 hd=64 showed 0 change). Watch ptxas.
# If 257+ → drops to 1 block/SM → catastrophe, revert.
#
# CRITERION:
#   +1-2% peak (568 → 575+) → mechanism worked on hd=128 (different from hd=64)
#   null (568.5 ± noise)    → short_scb structurally bound, all easy levers exhausted

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v98 hd=128 K-preload ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v98_kpreload_hd128_fp8_forward.cu \
    -o runs/fa_v98_kpreload -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -40

echo ""
echo "=== Reference points ==="
echo "  v96 (champion): 242 regs, 568.5T median at bh=64 sl=8192 (PEAK)"
echo "  v90 hd=64 analog: regs unchanged, perf null (compiler already optimal)"
echo "  v98 prediction: regs +0 to +8 likely. Peak +1-2% or null."
echo ""
echo "=== Run: correctness 8/8 + bench (small/mid/large/sliding) ==="
runs/fa_v98_kpreload
