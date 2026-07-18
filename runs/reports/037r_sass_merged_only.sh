#!/bin/bash
FULL=/data/lib/podman-data/projects/goml/runs/reports/037r_sass_wall_full.txt
OUT=/data/lib/podman-data/projects/goml/runs/reports/037r_sass_merged_only.txt
# Merged kernel_v1 = lines 34..5361
awk 'NR>=34 && NR<=5361' "$FULL" > "$OUT"
echo "=== merged_v1 SASS extracted, lines 34-5361 ==="
wc -l "$OUT"
echo ""
echo "-- STG per kernel_merged_v1 --"
grep -oE "STG\.[A-Za-z0-9.]+" "$OUT" | sort | uniq -c
echo "-- STS per kernel_merged_v1 --"
grep -oE "STS\.[A-Za-z0-9.]+" "$OUT" | sort | uniq -c
echo "-- LDS per kernel_merged_v1 --"
grep -oE "LDS\.[A-Za-z0-9.]+" "$OUT" | sort | uniq -c
echo "-- LDG per kernel_merged_v1 --"
grep -oE "LDG\.[A-Za-z0-9.]+" "$OUT" | sort | uniq -c
echo "-- Total lines --"
wc -l "$OUT"
