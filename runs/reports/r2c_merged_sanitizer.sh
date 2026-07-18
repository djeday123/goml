#!/bin/bash
export HOME=/tmp
SANITIZER=/usr/local/cuda-13.1/compute-sanitizer/compute-sanitizer
"$SANITIZER" --tool memcheck --launch-timeout 60 \
    /data/lib/podman-data/projects/goml/libs/r2c_merged_bit_exact 2>&1 | tail -5
