#!/bin/bash
export HOME=/tmp
cd /data/lib/podman-data/projects/goml/runs/reports
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -include string -include cuda_fp16.h -gencode arch=compute_120a,code=sm_120a 070_vram_probe.cu -o 070_vram_probe 2>&1 | tail -10
echo "done"
