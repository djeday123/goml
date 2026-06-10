#!/bin/bash
# Variance check: new v69 (with window param) vs old v69 (commit 665e9fd).
# 3 runs each, focused on bh=16 sl=4096 and bh=32 sl=2048 (the suspected jump configs).
# Capture nvidia-smi clocks to rule out boost-clock variance.

set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NVSMI=/usr/bin/nvidia-smi

cd "$GOML"

echo "=== GPU baseline state ==="
"$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics|SM|Memory|Boost" | head -8

echo ""
echo "=== Build OLD v69 (from commit 665e9fd, no window param) ==="
git -C "$GOML" show 665e9fd:libs/flash_attention_v69_fp8_forward.cu > /tmp/old_v69.cu
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    /tmp/old_v69.cu -o runs/fa_v69_old -lcudart 2>&1 | tail -3
echo "OLD v69 binary: $(ls -la runs/fa_v69_old | awk '{print $5}') bytes"

echo ""
echo "=== Rebuild NEW v69 (with window param) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    libs/flash_attention_v69_fp8_forward.cu \
    -o runs/fa_v69_fp8 -lcudart 2>&1 | tail -3

run_bench() {
    local LABEL=$1
    local BIN=$2
    local RUN_N=$3
    echo ""
    echo "--- $LABEL run $RUN_N ---"
    "$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics" | head -2 | sed 's/^/    PRE: /'
    "$BIN" 2>&1 | grep -E "bh=(16|32).*4096|bh=(16|32).*2048" | grep -v "wnd=1024"
    "$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics" | head -2 | sed 's/^/    POST: /'
}

echo ""
echo "######################################"
echo "# 3× OLD v69 (commit 665e9fd)"
echo "######################################"
for run in 1 2 3; do
    run_bench "OLD v69" runs/fa_v69_old $run
    sleep 1
done

echo ""
echo "######################################"
echo "# 3× NEW v69 (with window param)"
echo "######################################"
for run in 1 2 3; do
    run_bench "NEW v69" runs/fa_v69_fp8 $run
    sleep 1
done

echo ""
echo "=== Final GPU state ==="
"$NVSMI" -q -d CLOCK -i 0 | grep -E "Graphics|SM" | head -4
