#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
echo "=== dk_new ==="
"$NCU" --kernel-name kernel_dk_new --launch-count 1 \
    --metrics \
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active,\
dram__bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts.sum,\
smsp__inst_executed.sum \
    /data/lib/podman-data/projects/goml/libs/r1b_dk_wall 2>&1 | tail -20

echo ""
echo "=== dq_new ==="
"$NCU" --kernel-name kernel_dq_new --launch-count 1 \
    --metrics \
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
dram__bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts.sum,\
smsp__inst_executed.sum \
    /data/lib/podman-data/projects/goml/libs/r1c_dq_wall 2>&1 | tail -20
