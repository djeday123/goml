#!/bin/bash
# 037-r2: NCu-налог факт — пара измерений разных режимов
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/037r2_ncu_tax_data.txt
> "$OUT"

"$GATE" 2>&1 | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "=== Reference: чистый isolated wall (medium 5-run stand-protocol) ===" | tee -a "$OUT"
echo "27.836 ms (037-r шаг 6.0a, source: 037r_wall_data.txt)" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# NCu режим #1: single metric, 1 pass (dram-only)
echo "=== NCu режим A: dram__bytes.sum (1 pass) ===" | tee -a "$OUT"
$NCU --kernel-name kernel_merged_v1 --launch-count 1 \
     --metrics dram__bytes.sum "$BIN" 2>&1 | grep -E "avg_ms|dram__bytes" | tee -a "$OUT"

echo "" | tee -a "$OUT"

# NCu режим #2: multi-metric heavy stall (7 passes)
echo "=== NCu режим B: full stall breakdown (~7 passes) ===" | tee -a "$OUT"
$NCU --kernel-name kernel_merged_v1 --launch-count 1 \
     --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct \
     "$BIN" 2>&1 | grep -E "avg_ms|passes|mio_throttle|barrier|wait" | tee -a "$OUT"

echo "" | tee -a "$OUT"

# NCu режим #3: --set full (~40+ passes)
echo "=== NCu режим C: full set (all sections, максимум passes) ===" | tee -a "$OUT"
$NCU --kernel-name kernel_merged_v1 --launch-count 1 --set base \
     "$BIN" 2>&1 | grep -E "avg_ms|passes|Metric" | head -5 | tee -a "$OUT"
