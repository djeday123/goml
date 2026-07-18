#!/bin/bash
# 037-r2: SASS счёт инструкций на СВЕЖЕЙ сборке
GATE=/data/lib/podman-data/projects/goml/runs/reports/037r_gate.sh
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT_FULL=/data/lib/podman-data/projects/goml/runs/reports/037r2_sass_full.txt
OUT_MERGED=/data/lib/podman-data/projects/goml/runs/reports/037r2_sass_merged_only.txt
OUT_LOG=/data/lib/podman-data/projects/goml/runs/reports/037r2_sass_data.txt
> "$OUT_LOG"

"$GATE" 2>&1 | tee -a "$OUT_LOG"

/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" 2>&1 | tee "$OUT_FULL" > /dev/null

echo "" | tee -a "$OUT_LOG"
echo "== Functions in fresh binary ==" | tee -a "$OUT_LOG"
grep -nE "Function\s*:\s*_Z" "$OUT_FULL" | tee -a "$OUT_LOG"

# Extract kernel_merged_v1 range (from first Function line to next Function line)
START=$(grep -nE "Function\s*:.*kernel_merged_v1" "$OUT_FULL" | head -1 | cut -d: -f1)
END=$(grep -nE "Function\s*:" "$OUT_FULL" | awk -F: -v s="$START" '$1>s{print $1; exit}')
if [ -z "$END" ]; then END=$(wc -l < "$OUT_FULL"); fi
echo "kernel_merged_v1 SASS lines: [$START, $END)" | tee -a "$OUT_LOG"
awk -v s="$START" -v e="$END" 'NR>=s && NR<e' "$OUT_FULL" > "$OUT_MERGED"

echo "" | tee -a "$OUT_LOG"
echo "-- STG per kernel_merged_v1 --" | tee -a "$OUT_LOG"
grep -oE "STG\.[A-Za-z0-9.]+" "$OUT_MERGED" | sort | uniq -c | tee -a "$OUT_LOG"
echo "-- STS per kernel_merged_v1 --" | tee -a "$OUT_LOG"
grep -oE "STS\.[A-Za-z0-9.]+" "$OUT_MERGED" | sort | uniq -c | tee -a "$OUT_LOG"
echo "-- LDS per kernel_merged_v1 --" | tee -a "$OUT_LOG"
grep -oE "LDS\.[A-Za-z0-9.]+" "$OUT_MERGED" | sort | uniq -c | tee -a "$OUT_LOG"
echo "-- LDG per kernel_merged_v1 --" | tee -a "$OUT_LOG"
grep -oE "LDG\.[A-Za-z0-9.]+" "$OUT_MERGED" | sort | uniq -c | tee -a "$OUT_LOG"
echo "-- BAR per kernel_merged_v1 --" | tee -a "$OUT_LOG"
grep -cE "BAR\.SYNC|BAR\." "$OUT_MERGED" | tee -a "$OUT_LOG"
echo "-- SHFL per kernel_merged_v1 --" | tee -a "$OUT_LOG"
grep -cE "SHFL" "$OUT_MERGED" | tee -a "$OUT_LOG"
echo "-- CP.ASYNC per kernel_merged_v1 --" | tee -a "$OUT_LOG"
grep -oE "CP\.ASYNC[A-Za-z0-9._]*|LDGSTS\.[A-Za-z0-9.]+" "$OUT_MERGED" | sort | uniq -c | tee -a "$OUT_LOG"
