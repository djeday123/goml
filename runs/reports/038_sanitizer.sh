#!/bin/bash
# 038 sanitizer memcheck after E
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/038_sanitizer_data.txt
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck --error-exitcode 42 "$BIN" 2>&1 | tee "$OUT" | tail -20
echo "---"
echo "Exit code: ${PIPESTATUS[0]}"
