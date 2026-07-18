#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
METRICS="dram__bytes.sum,dram__bytes.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"

for K in kernel_merged_v1 kernel_dk_new kernel_dq_new kernel_d_precompute; do
  echo "===== $K ====="
  "$NCU" --kernel-name "$K" --launch-count 1 \
      --metrics "$METRICS" \
      "$BIN" 2>&1 | tail -12
done
