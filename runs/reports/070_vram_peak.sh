#!/bin/bash
# Poll nvidia-smi tight loop while bench runs, capture max VRAM.
export HOME=/tmp
cd /data/lib/podman-data/projects/goml/libs

LOG=/data/lib/podman-data/projects/goml/runs/reports/070_vram_peak.txt
> "$LOG"

for MODE in CURRENT BASE; do
    echo "=== $MODE ===" | tee -a "$LOG"

    if [ "$MODE" = "BASE" ]; then
        cp bench_r2c_e2e.cu bench_r2c_e2e.cu.070_final    # save current 070 state
        cp bench_r2c_e2e.cu.pre_070 bench_r2c_e2e.cu
        make -f Makefile.bench_r2c_e2e clean >/dev/null 2>&1
        make -f Makefile.bench_r2c_e2e >/dev/null 2>&1
    fi

    # Warm up device
    ./bench_r2c_e2e >/dev/null 2>&1
    sleep 2

    # Baseline nvidia-smi
    BASE_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    # Launch bench in background
    ./bench_r2c_e2e >/dev/null 2>&1 &
    BPID=$!

    # Tight-poll for peak
    MAX_MB=0
    for i in $(seq 1 200); do
        MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "$MB" -gt "$MAX_MB" ]; then MAX_MB=$MB; fi
        if ! kill -0 $BPID 2>/dev/null; then break; fi
        sleep 0.05
    done
    wait $BPID 2>/dev/null

    DELTA=$((MAX_MB - BASE_MB))
    echo "$MODE: base=${BASE_MB} MB, peak=${MAX_MB} MB, delta=${DELTA} MB" | tee -a "$LOG"

    if [ "$MODE" = "BASE" ]; then
        # restore 070 final state
        cp bench_r2c_e2e.cu.070_final bench_r2c_e2e.cu
        make -f Makefile.bench_r2c_e2e clean >/dev/null 2>&1
        make -f Makefile.bench_r2c_e2e >/dev/null 2>&1
    fi
done

echo "done" | tee -a "$LOG"
