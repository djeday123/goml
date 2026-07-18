#!/bin/bash
# 010 step-0: SASS audit dk_new + dq_new (measure-only).
# Uses CUDA 13.1 cuobjdump/nvdisasm for sm_120a SASS.

CUDA=/usr/local/cuda-13.1
export PATH=$CUDA/bin:$PATH

OUTDIR=/data/lib/podman-data/projects/goml/runs/reports
BIN_DK=/data/lib/podman-data/projects/goml/libs/r1b_dk_wall
BIN_DQ=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall

echo "=== TOOLS ==="
cuobjdump --version 2>&1 | head -2
nvdisasm --version 2>&1 | head -2

echo ""
echo "=== SASS dk_new ==="
cuobjdump --dump-sass "$BIN_DK" 2>&1 | grep -A 999999 "kernel_dk_new" | head -600 > "$OUTDIR/010_dk_new_sass.txt"
wc -l "$OUTDIR/010_dk_new_sass.txt"

echo ""
echo "=== SASS dq_new ==="
cuobjdump --dump-sass "$BIN_DQ" 2>&1 | grep -A 999999 "kernel_dq_new" | head -600 > "$OUTDIR/010_dq_new_sass.txt"
wc -l "$OUTDIR/010_dq_new_sass.txt"

echo ""
echo "=== SASS COUNT-STATS dk_new ==="
grep -c -E "^[[:space:]]+/\*[0-9a-fA-F]+\*/[[:space:]]+LDSM" "$OUTDIR/010_dk_new_sass.txt" | head -1 | xargs -I{} echo "LDSM (ldmatrix): {}"
grep -c -E "LDS\.128" "$OUTDIR/010_dk_new_sass.txt" | xargs -I{} echo "LDS.128: {}"
grep -c -E "LDS\.64" "$OUTDIR/010_dk_new_sass.txt" | xargs -I{} echo "LDS.64: {}"
grep -c -E "LDS\.32" "$OUTDIR/010_dk_new_sass.txt" | xargs -I{} echo "LDS.32: {}"
grep -c -E "LDS " "$OUTDIR/010_dk_new_sass.txt" | xargs -I{} echo "LDS (any): {}"
grep -c -E "HMMA|IMMA|MMA" "$OUTDIR/010_dk_new_sass.txt" | xargs -I{} echo "MMA-any: {}"
grep -c -E "STS" "$OUTDIR/010_dk_new_sass.txt" | xargs -I{} echo "STS: {}"

echo ""
echo "=== SASS COUNT-STATS dq_new ==="
grep -c -E "LDSM" "$OUTDIR/010_dq_new_sass.txt" | xargs -I{} echo "LDSM: {}"
grep -c -E "LDS\.128" "$OUTDIR/010_dq_new_sass.txt" | xargs -I{} echo "LDS.128: {}"
grep -c -E "LDS\.64" "$OUTDIR/010_dq_new_sass.txt" | xargs -I{} echo "LDS.64: {}"
grep -c -E "LDS\.32" "$OUTDIR/010_dq_new_sass.txt" | xargs -I{} echo "LDS.32: {}"
grep -c -E "LDS " "$OUTDIR/010_dq_new_sass.txt" | xargs -I{} echo "LDS (any): {}"
grep -c -E "HMMA|IMMA|MMA" "$OUTDIR/010_dq_new_sass.txt" | xargs -I{} echo "MMA-any: {}"
grep -c -E "STS" "$OUTDIR/010_dq_new_sass.txt" | xargs -I{} echo "STS: {}"
