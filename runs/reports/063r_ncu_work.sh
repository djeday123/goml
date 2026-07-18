#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e
LOG=/data/lib/podman-data/projects/goml/runs/reports/063r_ncu_work.txt
> "$LOG"

# Работа-детектор: executed instructions + MMA/QMMA + DRAM + wavefronts.
# Сравниваем causal vs nc: при честном скипе MMA_causal ~= 0.52-0.55 × MMA_nc
WORK="smsp__inst_executed.sum,sm__inst_executed_pipe_tensor_op_hmma.sum,dram__bytes.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,launch__grid_size"

for MODE in NC CAUSAL; do
    if [ "$MODE" = "NC" ]; then ENV_CAUSAL="0"; else ENV_CAUSAL="1"; fi
    echo "=========================================" | tee -a "$LOG"
    echo "===== MODE=$MODE (CAUSAL=$ENV_CAUSAL) =====" | tee -a "$LOG"
    echo "=========================================" | tee -a "$LOG"
    for KERNEL in kernel_merged_v1 kernel_dk_new kernel_dq_new kernel_d_precompute; do
        echo "----- $KERNEL -----" | tee -a "$LOG"
        env CAUSAL="$ENV_CAUSAL" "$NCU" --kernel-name $KERNEL --launch-count 1 --metrics "$WORK" "$BIN" 2>&1 | tail -12 | tee -a "$LOG"
    done
done
