#!/bin/bash
# TMA probe on sm_120a — two independent tests, isolated binaries.
#   [A] cp.async.bulk + mbarrier (sm_90 bulk primitive)
#   [B] cp.async.bulk.tensor.2d  (real TMA via tensormap)
# Each compiles + runs separately so a device-kill in one doesn't poison the other.
# Compile failure = ptxas rejects the family on sm_120a.
# Runtime failure = HW/firmware lacks the execution unit.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
NVCC="$CUDA/bin/nvcc"
NVCC_FLAGS="-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -lcuda"

cd "$GOML"

echo "============================================================"
echo "  [A] cp.async.bulk + mbarrier"
echo "============================================================"
echo "Compile:"
$NVCC $NVCC_FLAGS runs/tma_probe_A_bulk.cu -o runs/tma_probe_A 2>&1 | tail -10
CA=${PIPESTATUS[0]}
if [ "$CA" -ne 0 ]; then
    echo "[A] COMPILE-FAIL: ptxas does not accept cp.async.bulk on sm_120a"
else
    echo "[A] compile OK"
    echo ""
    echo "Run:"
    runs/tma_probe_A
fi

echo ""
echo "============================================================"
echo "  [B] cp.async.bulk.tensor.2d (real TMA)"
echo "============================================================"
echo "Compile:"
$NVCC $NVCC_FLAGS runs/tma_probe_B_tensor.cu -o runs/tma_probe_B 2>&1 | tail -10
CB=${PIPESTATUS[0]}
if [ "$CB" -ne 0 ]; then
    echo "[B] COMPILE-FAIL: ptxas does not accept cp.async.bulk.tensor on sm_120a"
else
    echo "[B] compile OK"
    echo ""
    echo "Run:"
    runs/tma_probe_B
fi

echo ""
echo "============================================================"
echo "  Verdict"
echo "============================================================"
echo "[A] compile=$([ $CA -eq 0 ] && echo OK || echo FAIL)"
echo "[B] compile=$([ $CB -eq 0 ] && echo OK || echo FAIL)"
echo "  → A=OK, B=OK         → real TMA available, address-arith reduction path is OPEN"
echo "  → A=OK, B=FAIL       → bulk works, no tensormap path on consumer Blackwell"
echo "  → A=FAIL             → no bulk family at all (very unlikely on sm_120a)"
