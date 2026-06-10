#!/bin/bash
# Full QMMA SASS injection experiment.
# Builds baseline cubin, dumps SASS, patches bit 76, runs both, compares.

set -euo pipefail

CUDA=/usr/local/cuda-13.1
RUNS=/data/lib/podman-data/projects/goml/runs

cd "$RUNS"

echo "=== Step 1: compile baseline kernel ==="
"$CUDA/bin/nvcc" -arch=sm_120a -cubin qmma_inject_kernel.cu -o qmma_baseline.cubin
echo "  -> qmma_baseline.cubin"

echo ""
echo "=== Step 2: confirm SASS contains QMMA.16816 ==="
"$CUDA/bin/cuobjdump" --dump-sass qmma_baseline.cubin | grep -E "QMMA|HMMA|OMMA" | head -20 || true

echo ""
echo "=== Step 3: locate and patch QMMA bytes (bit 76 → INVALID2) ==="
python3 qmma_patch.py qmma_baseline.cubin qmma_invalid2.cubin

echo ""
echo "=== Step 4: confirm patched SASS disassembles to INVALID2 ==="
"$CUDA/bin/cuobjdump" --dump-sass qmma_invalid2.cubin | grep -E "QMMA|HMMA|OMMA" | head -20 || true

echo ""
echo "=== Step 5: compile host driver ==="
"$CUDA/bin/nvcc" -O2 qmma_inject_host.cu -lcuda -o qmma_host

echo ""
echo "=== Step 6: run baseline ==="
./qmma_host qmma_baseline.cubin 256 1 1 || echo "[baseline FAILED]"

echo ""
echo "=== Step 7: run INVALID2-patched ==="
./qmma_host qmma_invalid2.cubin 256 1 1 || echo "[patched FAILED — likely INVALID_INSTRUCTION]"

echo ""
echo "=== Done. Compare 'C[0..7]' lines and 'issues/sec' between the two. ==="
