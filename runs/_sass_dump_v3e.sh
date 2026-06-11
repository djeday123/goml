#!/bin/bash
CUDA=/usr/local/cuda-13.1
BIN=/data/lib/podman-data/projects/goml/runs/mbar_repro_v3e
OUT=/data/lib/podman-data/projects/goml/runs/v3e.sass

echo "=== Full SASS dump ==="
"$CUDA/bin/cuobjdump" -sass "$BIN" > "$OUT" 2>&1
wc -l "$OUT"

echo ""
echo "=== Q1: barrier-related instructions in SASS ==="
grep -nE '\bBAR\.|\bATOMS\.|\bLDSM\b|\bMEMBAR\b|\bBARRIER\b' "$OUT" | head -30

echo ""
echo "=== Q2: BAR.SYNC / BAR.RED / BAR.ARV by barrier ID (look at first imm operand) ==="
grep -nE 'BAR\.SYNC|BAR\.RED|BAR\.ARV' "$OUT"

echo ""
echo "=== Q3: ATOMS = SMEM atomic (mbarrier compiled as atomic ops) ==="
grep -nE 'ATOMS\.' "$OUT" | head -20

echo ""
echo "=== Look for mbarrier-related stores/loads to SMEM ==="
grep -nE 'STS|LDS' "$OUT" | head -10
