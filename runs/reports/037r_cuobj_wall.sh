#!/bin/bash
# 037-r: cuobjdump stale binary opознание
OUT=/data/lib/podman-data/projects/goml/runs/reports/037r_wall_res.txt
> "$OUT"
echo "=== r2c_merged_wall (Jul 6 07:14) ===" | tee -a "$OUT"
/usr/local/cuda-13.1/bin/cuobjdump --dump-resource-usage \
    /data/lib/podman-data/projects/goml/libs/r2c_merged_wall 2>&1 | tee -a "$OUT" | grep -A5 merged_v1
