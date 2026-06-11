#!/bin/bash
OUT=/data/lib/podman-data/projects/goml/runs/v3e.sass

echo "=== All SYNCS.* opcodes (mbarrier compiled to SYNCS on Blackwell) ==="
grep -nE 'SYNCS\.' "$OUT"

echo ""
echo "=== All distinct full opcodes used ==="
grep -oE '\s[@!P0-9 ]*[A-Z][A-Z0-9._]+' "$OUT" | \
    awk '{print $NF}' | sort -u | head -60

echo ""
echo "=== Look at context around SYNCS.ARRIVE ==="
grep -B3 -A1 'SYNCS\.ARRIVE' "$OUT" | head -30

echo ""
echo "=== Look at context around SYNCS.* generally ==="
grep -B2 -A2 'SYNCS\.' "$OUT" | head -80
