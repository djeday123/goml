#!/bin/bash
# Build FP8 GEMM shared library for GoML v3
set -e

CUDA_DIR="${CUDA_DIR:-libs}"
GPU_ARCH="${GPU_ARCH:-sm_89}"
LIB="libfp8gemm.so"

echo "=== Building FP8 GEMM ==="
echo "Source: ${CUDA_DIR}/fp8_gemm.cu"
echo "Target: ${GPU_ARCH}"

# Shared library
nvcc -O3 -arch=${GPU_ARCH} -std=c++17 \
    --shared -Xcompiler -fPIC \
    ${CUDA_DIR}/fp8_gemm.cu \
    -o ${CUDA_DIR}/${LIB}

ls -lh ${CUDA_DIR}/${LIB}

# Check symbols
echo ""
echo "=== Exported symbols ==="
nm -D ${CUDA_DIR}/${LIB} | grep " T " | grep fp8

# Benchmark binary (optional)
if [ "$1" = "bench" ]; then
    echo ""
    echo "=== Building benchmark ==="
    nvcc -O3 -arch=${GPU_ARCH} -std=c++17 \
        ${CUDA_DIR}/fp8_gemm.cu ${CUDA_DIR}/fp8_gemm_bench.cu \
        -o runs/fp8_gemm_bench -lcudart
    echo "Run: runs/fp8_gemm_bench"
fi

echo ""
echo "=== Install ==="
echo "  export LD_LIBRARY_PATH=\$(pwd)/${CUDA_DIR}:\$LD_LIBRARY_PATH"
echo "  # or: sudo cp ${CUDA_DIR}/${LIB} /usr/local/lib && sudo ldconfig"