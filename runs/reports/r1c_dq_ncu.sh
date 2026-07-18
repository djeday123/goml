#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
"$NCU" --kernel-name kernel_dq_new --launch-count 1 \
    --set full --print-summary per-kernel \
    /data/lib/podman-data/projects/goml/libs/r1c_dq_wall 2>&1 | tail -120
