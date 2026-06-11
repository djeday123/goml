#!/bin/bash
CUDA=/usr/local/cuda-13.1
BIN=/data/lib/podman-data/projects/goml/runs/fa_v96b_localfix
echo "=== HADD2 SASS occurrences (sample 20) ==="
"$CUDA/bin/cuobjdump" -sass "$BIN" | grep -E 'HADD2' | head -20
echo ""
echo "=== HMUL2 sample ==="
"$CUDA/bin/cuobjdump" -sass "$BIN" | grep -E 'HMUL2' | head -10
echo ""
echo "=== Context around first few HADD2 ==="
"$CUDA/bin/cuobjdump" -sass "$BIN" | grep -B2 -A1 'HADD2' | head -40
