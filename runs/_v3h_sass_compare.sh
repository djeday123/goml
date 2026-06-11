#!/bin/bash
CUDA=/usr/local/cuda-13.1
OUT=/data/lib/podman-data/projects/goml/runs/v3h.sass
"$CUDA/bin/cuobjdump" -sass /data/lib/podman-data/projects/goml/runs/mbar_repro_v3h > "$OUT" 2>&1
wc -l "$OUT"

echo ""
echo "=== All FENCE / MEMBAR instructions ==="
grep -nE 'FENCE|MEMBAR' "$OUT"

echo ""
echo "=== Context around SYNCS.ARRIVE.TRANS64 (1f40) ==="
awk '/1f40/{for(i=NR-5;i<=NR+3;i++) print; exit}' "$OUT" 2>/dev/null || sed -n '/1f40/,/1f60/p' "$OUT" | head -10
echo "---"
grep -B5 -A2 'SYNCS\.ARRIVE\.TRANS64' "$OUT"

echo ""
echo "=== Context around SYNCS.PHASECHK.TRANS64 ==="
grep -B5 -A2 'SYNCS\.PHASECHK' "$OUT"

echo ""
echo "=== v3e SASS compare (same regions) ==="
V3E=/data/lib/podman-data/projects/goml/runs/v3e.sass
echo "v3e FENCE/MEMBAR count:"
grep -cE 'FENCE|MEMBAR' "$V3E"
echo "v3h FENCE/MEMBAR count:"
grep -cE 'FENCE|MEMBAR' "$OUT"

echo ""
echo "=== v3e ARRIVE context ==="
grep -B5 -A2 'SYNCS\.ARRIVE\.TRANS64' "$V3E" | head -20

echo ""
echo "=== v3e PHASECHK context ==="
grep -B5 -A2 'SYNCS\.PHASECHK' "$V3E" | head -20
