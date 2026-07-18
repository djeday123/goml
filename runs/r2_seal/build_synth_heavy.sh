#!/bin/bash
/usr/local/cuda-13.1/bin/nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas=-v \
    /data/lib/podman-data/projects/goml/libs/t2_synth_residency.cu \
    -o /data/lib/podman-data/projects/goml/libs/t2_synth_heavy 2>&1
