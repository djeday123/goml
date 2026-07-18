#!/bin/bash
# 046 I: NCu bh=1 b2b режим
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_bh1_b2b
OUT=/data/lib/podman-data/projects/goml/runs/reports/046_ncu_b2b_data.txt
> "$OUT"
MET="dram__bytes.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active"
for k in kernel_merged_v1 kernel_dk_new kernel_dq_new; do
    echo "=== 046 I: NCu $k [bh=1 b2b + persist, NCu-mode] ===" | tee -a "$OUT"
    "$NCU" --kernel-name "$k" --launch-count 1 --metrics "$MET" "$BIN" 2>&1 | tail -12 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done
