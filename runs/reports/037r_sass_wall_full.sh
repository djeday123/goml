#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/037r_sass_wall_full.txt
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" 2>&1 | tee "$OUT" > /dev/null
echo "== Functions =="
grep -oE "Function.*_Z[A-Za-z0-9_]+" "$OUT" | sort -u
echo ""
echo "== kernel_merged_v1 line range =="
grep -n "Function.*kernel_merged_v1\|Function.*kernel_" "$OUT"
