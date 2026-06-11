#!/bin/bash
# Production matrix after localfix: build v96b + v113b, verify, run all kernels.

set -uo pipefail
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

build_kernel() {
    local src="$1" bin="$2"
    "$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
        -Xptxas=-v -lineinfo "$src" -o "$bin" -lcudart 2>&1 \
        | grep -E "stack|register|spill|barriers"
}

echo "================================================================"
echo "# Build v96b (localfix on v96 hd=128 champion)"
echo "================================================================"
build_kernel libs/flash_attention_v96b_localfix_hd128_fp8_forward.cu runs/fa_v96b_localfix

echo ""
echo "================================================================"
echo "# Build v113b (localfix on v113 niche)"
echo "================================================================"
build_kernel libs/flash_attention_v113b_localfix_hd128_fp8_forward.cu runs/fa_v113b_localfix

echo ""
echo "================================================================"
echo "# SASS LDL/STL check (all production-candidate kernels)"
echo "================================================================"
for b in fa_v96_ksbatched fa_v96b_localfix fa_v111_producer_skip fa_v118_localfix \
         fa_v113_producer_arrive fa_v113b_localfix; do
    if [ -x "runs/$b" ]; then
        local_count=$("$CUDA/bin/cuobjdump" -sass "runs/$b" 2>/dev/null | grep -cE '\bLDL\b|\bSTL\b')
        printf "  %-30s LDL+STL=%d\n" "$b" "$local_count"
    fi
done

echo ""
echo "================================================================"
echo "# Run v96b correctness + perf"
echo "================================================================"
runs/fa_v96b_localfix | sed -n '/Correctness/,$p'

echo ""
echo "================================================================"
echo "# Run v113b correctness + perf"
echo "================================================================"
runs/fa_v113b_localfix | sed -n '/Correctness/,$p'
