#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/runs/probes/fa_probe_bank
for N in 1 1000 1000000; do
  echo "--- P4 N=$N ---"
  "$NCU" --kernel-name "probe_kernel" --launch-count 1 \
    --metrics smsp__inst_executed_op_shared_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    "$BIN" 4 $N 2>&1 | grep -E "conflict|shared_ld"
done
