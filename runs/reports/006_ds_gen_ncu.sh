#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
"$NCU" --kernel-name kernel_ds_gen --launch-count 1 \
    --set full --print-summary per-kernel \
    /data/lib/podman-data/projects/goml/libs/r1a_wall 128 8192 0 5 5 2>&1 | tail -160
