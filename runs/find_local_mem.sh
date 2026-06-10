#!/bin/bash
# Find local-memory access sites in v68 SASS.
# LDL/STL = local memory load/store. They're the source of the 18.21%
# L1TEX local mem accesses NCu reported.
set -euo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Full SASS dump ==="
"$CUDA/bin/cuobjdump" --dump-sass runs/fa_v68_fp8 > runs/v68_sass.txt 2>&1
wc -l runs/v68_sass.txt

echo ""
echo "=== Count LDL/STL instructions per opcode ==="
grep -oE "LDL|STL" runs/v68_sass.txt | sort | uniq -c

echo ""
echo "=== First 40 LDL/STL with context (line before for nearby labels) ==="
grep -nE "LDL|STL" runs/v68_sass.txt | head -40 > runs/v68_local_mem.txt
cat runs/v68_local_mem.txt
