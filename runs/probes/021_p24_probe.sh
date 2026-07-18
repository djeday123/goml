#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/runs/probes/fa_probe_bank
N=1000000

for P in 24 25; do
  echo "=== Pattern P${P} (packed-STS.32 @68 + π_V, hk=$((P-24)), ks/s0=0) ==="
  "$NCU" --kernel-name "probe_kernel" --launch-count 1 \
    --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum \
    "$BIN" $P $N 2>&1 | tail -12
  echo ""
done
