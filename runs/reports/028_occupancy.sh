#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall
"$NCU" --kernel-name kernel_dq_new --launch-count 1 \
    --section Occupancy \
    --section LaunchStats \
    "$BIN" 2>&1 | grep -iE "Grid Size|Block Size|Threads|Achieved|Waves|Warps|SMs Idle|Est\.|Occupancy" | head -30
