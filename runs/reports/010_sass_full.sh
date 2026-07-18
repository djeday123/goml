#!/bin/bash
CUDA=/usr/local/cuda-13.1
OUTDIR=/data/lib/podman-data/projects/goml/runs/reports
BIN_DK=/data/lib/podman-data/projects/goml/libs/r1b_dk_wall
BIN_DQ=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall

# Full SASS dump (no head cutoff)
$CUDA/bin/cuobjdump --dump-sass "$BIN_DK" > "$OUTDIR/010_dk_new_sass_full.txt" 2>&1
$CUDA/bin/cuobjdump --dump-sass "$BIN_DQ" > "$OUTDIR/010_dq_new_sass_full.txt" 2>&1

echo "=== FILE SIZES ==="
wc -l "$OUTDIR/010_dk_new_sass_full.txt" "$OUTDIR/010_dq_new_sass_full.txt"

echo ""
echo "=== SAMPLE dk_new SASS lines (first 30) ==="
head -30 "$OUTDIR/010_dk_new_sass_full.txt"
