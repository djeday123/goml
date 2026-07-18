#!/bin/bash
# W1.0: измерить фактический wave tail через пары grid 376 / 512 / 752 / 1024.
# Одинаковая sl (одинаковая работа на блок), меняем bh для получения нужного grid.
# Wave = 376 на sm_120a (2 blocks/SM × 188 SMs).
set -uo pipefail
BIN=/data/lib/podman-data/projects/goml/runs/fa_v121r_diet
echo "=== W1.0 wave-tail measurement ==="
echo "Wave size = 376 (2 blocks/SM × 188 SMs)"
echo ""
echo "--- sl=1024 (nqt=8) work pairs ---"
"$BIN" --time 47 1024 0 30  # grid=376 (exact 1 wave)
"$BIN" --time 64 1024 0 30  # grid=512 (1.36 waves, last 36% util)
"$BIN" --time 94 1024 0 30  # grid=752 (exact 2 waves)
"$BIN" --time 128 1024 0 30 # grid=1024 (2.72 waves, last 73% util)
"$BIN" --time 188 1024 0 30 # grid=1504 (4 waves exact)
echo ""
echo "--- sl=2048 (nqt=16) work pairs ---"
"$BIN" --time 24 2048 0 30  # grid=384 (≈1 wave)
"$BIN" --time 32 2048 0 30  # grid=512
"$BIN" --time 47 2048 0 30  # grid=752
"$BIN" --time 64 2048 0 30  # grid=1024
"$BIN" --time 94 2048 0 30  # grid=1504
echo ""
echo "--- sl=4096 (nqt=32) work pairs ---"
"$BIN" --time 12 4096 0 30  # grid=384 (≈1 wave)
"$BIN" --time 16 4096 0 30  # grid=512
"$BIN" --time 24 4096 0 30  # grid=768
"$BIN" --time 32 4096 0 30  # grid=1024
"$BIN" --time 47 4096 0 30  # grid=1504
"$BIN" --time 64 4096 0 30  # grid=2048
echo ""
echo "Done."
