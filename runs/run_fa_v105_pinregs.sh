#!/bin/bash
# v105 = v104 + P-in-regs (port v89 mechanism). Pf_pair packed + shfl-gather for PV B.
# Attack v104's mio_throttle 21.36% + barrier 13.76% by removing smP STS+LDS + 2 syncs.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

echo "=== Build v105 hd=128 Br=96 6-warp P-in-regs ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v105_pinregs_br96_fp8_forward.cu \
    -o runs/fa_v105_pinregs -lcudart 2>&1 | grep -E "(register|spill|stack|smem|Function|warning|error)" | head -30

echo ""
echo "=== References ==="
echo "  v96  : 242 regs, 49.7 KB, 2 blocks/SM (8 warps), 568T peak"
echo "  v102 : 168 regs + 212B spill, 32 KB, 3 blocks (12 warps), 553T peak"
echo "  v104 : 156 regs no spill, 28 KB, 2 blocks (12 warps), 511T peak"
echo "  v104 stalls @ peak: wait 23%, mio 21.4%, barrier 13.8%, math_pipe 12.5%"
echo "  v105 target: ~165 regs no spill, mio + barrier should drop"
echo ""
echo "=== Run: perf only (correctness will fail inherited from v102 layout) ==="
runs/fa_v105_pinregs
