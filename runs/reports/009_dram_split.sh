#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics \
dram__bytes.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    /data/lib/podman-data/projects/goml/libs/r2c_merged_wall 2>&1 | tail -20
