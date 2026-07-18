#!/bin/bash
OUT_DIR=/data/lib/podman-data/projects/goml/runs/r2_seal
/usr/local/cuda-13.1/bin/nvcc -arch=sm_120a -ptx \
  /data/lib/podman-data/projects/goml/libs/mma_f16acc_probe.cu \
  -o $OUT_DIR/aa0_mma_f16acc.ptx 2>&1 | tee $OUT_DIR/aa0_ptx.log
