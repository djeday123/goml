#!/bin/bash
# Long-running v96 kernel for thermal/clock monitoring.
#
# Usage:
#   bash run_fa_v96_thermal.sh [seconds]   (default 180 = 3 min)
#
# THIS TERMINAL: rebuilds v96 with --loop mode, runs cfg=9 peak for N seconds.
# OTHER TERMINAL: run nvidia-smi monitor (commands printed below).

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

SECONDS_RUN=${1:-180}  # default 3 min

cd "$GOML"

echo "=== Build v96 with --loop mode ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v96_ksbatched_hd128_fp8_forward.cu \
    -o runs/fa_v96_ksbatched -lcudart 2>&1 | grep -E "(register|error)" | head -3

echo ""
echo "================================================================"
echo "  Open another terminal and run ONE of these to monitor:"
echo ""
echo "  Option 1 — nvidia-smi dmon (concise streaming):"
echo "    nvidia-smi dmon -s pucmt -d 1"
echo ""
echo "    columns: power(W), util%, mem-util%, clk_sm(MHz), clk_mem(MHz), temp(C)"
echo ""
echo "  Option 2 — full nvidia-smi every 0.5s:"
echo "    watch -n 0.5 'nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,clocks_throttle_reasons.active --format=csv,noheader'"
echo ""
echo "  Option 3 — log to file:"
echo "    nvidia-smi dmon -s pucmt -d 1 -o T > thermal_log.csv &"
echo "================================================================"
echo ""
echo "=== Running v96 cfg=9 (bh=64 sl=8192 hd=128) for ${SECONDS_RUN} seconds ==="
echo "    (cfg=9 is PEAK config, runs ~564T sustained)"
echo ""
runs/fa_v96_ksbatched --loop 9 ${SECONDS_RUN}

echo ""
echo "Done. If you logged thermal data, check it now."
