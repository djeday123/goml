#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall
OUT=/data/lib/podman-data/projects/goml/runs/probes/dq_sass_full.txt
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" > "$OUT" 2>&1
echo "Dumped to $OUT ($(wc -l < $OUT) lines)"
grep -n "Function : " "$OUT" | head -10
