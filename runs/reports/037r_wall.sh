#!/bin/bash
# 037-r wall isolated stand-protocol (post-cut merged, gate-verified 254r)
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
LOG=/data/lib/podman-data/projects/goml/runs/reports/037r_wall_data.txt
> "$LOG"

# Gate before measuring
"$GATE" >> "$LOG" 2>&1 || { echo "gate failed" >&2; exit 1; }

echo "=== WARMUP (4 runs, discarded) ==="
for i in 1 2 3 4; do "$BIN" 2>&1 | grep avg_ms > /dev/null; done
echo "=== MEASURED (5 runs, isolated wall, no NCu) ===" | tee -a "$LOG"
for i in 1 2 3 4 5; do
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep avg_ms)
    echo "run=$i temp=${TEMP}C mode=isolated $LINE" | tee -a "$LOG"
done
