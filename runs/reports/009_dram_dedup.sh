#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
echo "=== merged ==="
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics dram__bytes.sum,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,dram__sectors_read.sum,dram__sectors_write.sum \
    /data/lib/podman-data/projects/goml/libs/r2c_merged_wall 2>&1 | tail -25
echo ""
echo "=== ds_gen (R1) ==="
"$NCU" --kernel-name kernel_ds_gen --launch-count 1 \
    --metrics dram__bytes.sum,dram__sectors_read.sum,dram__sectors_write.sum \
    /data/lib/podman-data/projects/goml/libs/r1a_wall 128 8192 0 5 5 2>&1 | tail -12
echo ""
echo "=== dV_p1 (R1) ==="
"$NCU" --kernel-name kernel_dv_mma_p1 --launch-count 1 \
    --metrics dram__bytes.sum,dram__sectors_read.sum,dram__sectors_write.sum \
    /data/lib/podman-data/projects/goml/libs/bench_r1_e2e 2>&1 | tail -12
