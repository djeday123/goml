#!/bin/bash
# Step 2 pre-checks (a, b, c) before v68 implementation.
# Outputs go to runs/v68_precheck_*.txt for review.

set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "##############################"
echo "# Pre-check 2b: SMEM caps"
echo "##############################"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/check_smem_caps.cu -o runs/check_smem_caps -lcudart
runs/check_smem_caps | tee runs/v68_precheck_smem.txt

echo ""
echo "##############################"
echo "# Pre-check 2c: ldmatrix.trans B-frag probe"
echo "##############################"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/ldsm_trans_probe.cu -o runs/ldsm_trans_probe -lcudart
runs/ldsm_trans_probe | tee runs/v68_precheck_ldsm.txt

echo ""
echo "##############################"
echo "# Pre-check 2a: source-attributed bank conflict location"
echo "# (separate script — slow, ~3-5 min)"
echo "##############################"
echo "Run separately:"
echo "    bash runs/run_ncu_v66_source.sh"
