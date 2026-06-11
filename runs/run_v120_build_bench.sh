#!/bin/bash
# v120a (half-phase 1000ns) + v120b (quarter-phase 500ns) build + bench ×3 vs v96b same thermal.
set -uo pipefail
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"

build_one() {
    local src="$1" bin="$2"
    "$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
        -Xptxas=-v -lineinfo "$src" -o "$bin" -lcudart 2>&1 \
        | grep -E "(register|spill|stack|error)" | head -5
}

echo "=== Build v96b (baseline) ==="
build_one libs/flash_attention_v96b_localfix_hd128_fp8_forward.cu runs/fa_v96b_localfix
echo ""
echo "=== Build v120a (half-phase 1000ns) ==="
build_one libs/flash_attention_v120a_halfphase_hd128_fp8_forward.cu runs/fa_v120a_halfphase
echo ""
echo "=== Build v120b (quarter-phase 500ns) ==="
build_one libs/flash_attention_v120b_quarterphase_hd128_fp8_forward.cu runs/fa_v120b_quarterphase
echo ""

# Bench ×3 ровно по одному и тому же sequence — same thermal.
# Берём cfg=9 (peak bh=64 sl=8192) через --loop 9 N или просто полный bench output.
# Полный bench внутри v96b/v120 main() — он печатает 23 строки с TFLOPS,
# что и есть бенч ×3 (median of 3 launches per row).

run_bench() {
    local bin="$1" tag="$2"
    echo "================================"
    echo "  $tag — bench (full main)"
    echo "================================"
    # Полный main печатает correctness + bench.
    # Берём только bench-секцию.
    "$bin" 2>&1 | awk '/--- Performance ---/{p=1} p{print}'
}

run_bench runs/fa_v96b_localfix       "v96b baseline"
echo ""
run_bench runs/fa_v120a_halfphase     "v120a half-phase 1000ns"
echo ""
run_bench runs/fa_v120b_quarterphase  "v120b quarter-phase 500ns"
