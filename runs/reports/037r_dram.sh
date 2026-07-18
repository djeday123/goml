#!/bin/bash
# 037-r DRAM discriminator: ожидание post-cut ~9.8 GB per launch (033-c)
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/037r_dram_data.txt
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    "$BIN" 2>&1 | tail -20 | tee "$OUT"
