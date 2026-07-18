#!/bin/bash
# 050 §3.c racecheck (правило 13: барьер line 310 умер, обязателен)
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/050_racecheck_data.txt
/usr/local/cuda-13.1/bin/compute-sanitizer --tool racecheck --error-exitcode 42 "$BIN" 2>&1 | tee "$OUT" | tail -5
echo "Exit code: ${PIPESTATUS[0]}"
