#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/runs/probes/transpose_ds_unit_test
OUT=/data/lib/podman-data/projects/goml/runs/probes/transpose_ds_sass.txt
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" > "$OUT" 2>&1
echo "=== SASS gates ==="
echo -n "SHFL: "; grep -c SHFL "$OUT"
echo -n "STS (any): "; grep -cE "^\s+/\*[0-9a-f]+\*/\s+STS\b" "$OUT"
echo -n "STS.U8/16: "; grep -cE "STS\.(U8|U16|B8|B16)" "$OUT"
echo -n "LDL/STL: "; grep -cE "\bLDL\b|\bSTL\b" "$OUT"
echo -n "PRMT: "; grep -c PRMT "$OUT"
echo -n "LDS: "; grep -cE "^\s+/\*[0-9a-f]+\*/\s+LDS\b" "$OUT"
