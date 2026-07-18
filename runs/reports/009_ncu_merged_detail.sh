#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
"$NCU" --kernel-name kernel_merged_v1 --launch-count 1 \
    --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts.sum,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_membar_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__inst_executed.sum,\
smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
smsp__inst_executed_op_ldsm.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_bytes_equiv_l1sectormiss.sum \
    /data/lib/podman-data/projects/goml/libs/r2c_merged_wall 2>&1 | tail -30
