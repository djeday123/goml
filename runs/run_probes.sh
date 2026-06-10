#!/bin/bash
# Probe each PTX feature on sm_120a; report SUPPORTED / not supported.
cd /data/lib/podman-data/projects/goml
for P in TMA_BULK STMATRIX_X4 STMATRIX_X4_TRANS MBARRIER FP8_MMA FP6_MMA FP4_MMA; do
  echo "=== $P on sm_120a ==="
  out=$(env PATH=/usr/local/cuda-13.1/bin:/usr/bin nvcc \
        -gencode arch=compute_120a,code=sm_120a \
        -DPROBE_$P -O3 runs/sm120_capabilities.cu \
        -o /tmp/probe_test 2>&1)
  if [ -f /tmp/probe_test ]; then
    echo "  -> SUPPORTED"
    rm -f /tmp/probe_test
  else
    echo "$out" | grep -E "error|not supported" | head -3
    echo "  -> ERRORS"
  fi
done
