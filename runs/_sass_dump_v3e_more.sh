#!/bin/bash
OUT=/data/lib/podman-data/projects/goml/runs/v3e.sass
echo "=== MBAR* opcodes ==="
grep -nE 'MBAR|MB\.' "$OUT" | head -30
echo ""
echo "=== ATOMS / ATOMG / RED ==="
grep -nE 'ATOMS|ATOMG|\bRED\b' "$OUT" | head -20
echo ""
echo "=== ULDC.64 (uniform constant load) near barriers ==="
grep -nE 'ULDC\.|ATOMS\.CAS|CCTLT' "$OUT" | head -20
echo ""
echo "=== All distinct uppercase opcodes used ==="
grep -oE '^\s*/\*[0-9a-f]+\*/\s+[@!P0-9 ]*[A-Z][A-Z0-9._]+' "$OUT" \
    | sed -E 's/.*\s([A-Z][A-Z0-9._]+).*/\1/' | sort -u | head -50
echo ""
echo "=== Look for what's near consumer mbar_wait (test_wait.parity loop) ==="
echo "Searching for SETP or @!P pattern around line 1300-1380:"
sed -n '1280,1380p' "$OUT" | head -60
