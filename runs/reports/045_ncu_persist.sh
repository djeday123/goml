#!/bin/bash
# 045 I: NCu bh=1 sl=8192 с persist L2 window на dS_nat
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_bh1_persist
OUT=/data/lib/podman-data/projects/goml/runs/reports/045_ncu_persist_data.txt
> "$OUT"

MET_DRAM="dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active"

for k in kernel_merged_v1 kernel_dk_new kernel_dq_new; do
    echo "=== 045 I: NCu $k [bh=1 persist window, NCu-mode] ===" | tee -a "$OUT"
    "$NCU" --kernel-name "$k" --launch-count 1 --metrics "$MET_DRAM" "$BIN" 2>&1 | tail -15 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done
