#!/bin/bash
NVCC=/usr/local/cuda-13.1/bin/nvcc
SRCS="/data/lib/podman-data/projects/goml/libs/r1b_dk_wall.cu /data/lib/podman-data/projects/goml/libs/fa_bwd_ds_gen.cu /data/lib/podman-data/projects/goml/libs/fa_bwd_dk.cu /data/lib/podman-data/projects/goml/libs/fa_bwd_dk_new.cu"
for R in 126 124; do
  echo "===== maxrregcount=$R ====="
  $NVCC -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas=-v -maxrregcount $R $SRCS -o /data/lib/podman-data/projects/goml/libs/r1b_dk_${R} 2>&1 | grep -A2 "kernel_dk_new" | tail -3
  for i in 1 2 3; do
    /data/lib/podman-data/projects/goml/libs/r1b_dk_${R} 2>&1 | grep avg_ms
  done
done
