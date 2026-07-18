#!/bin/bash
# 044 I.3.b: NCu per-kernel bh=1 sl=8192
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_bh1_sl8192
OUT=/data/lib/podman-data/projects/goml/runs/reports/044_ncu_bh1_data.txt
> "$OUT"

MET_DRAM="dram__bytes.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active"

for k in kernel_merged_v1 kernel_dk_new kernel_dq_new; do
    echo "=== 044 I.3.b: NCu $k [bh=1, режим NCu-mode] ===" | tee -a "$OUT"
    "$NCU" --kernel-name "$k" --launch-count 1 --metrics "$MET_DRAM" "$BIN" 2>&1 | tail -12 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done
