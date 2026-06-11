#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    runs/setmaxnreg_probe.cu -o runs/setmaxnreg_probe -lcudart 2>&1 | grep -E "error|warning"
rc_build=$?
echo "build_rc=$rc_build"
if [ -x runs/setmaxnreg_probe ]; then
    echo "=== Run ==="
    runs/setmaxnreg_probe
    echo "exec_rc=$?"
else
    echo "(binary not built)"
fi
