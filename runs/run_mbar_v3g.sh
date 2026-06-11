#!/bin/bash
CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
cd "$GOML"
echo "=== Build v3g (bar.sync 1 → bar.sync 7) ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v runs/mbar_repro_v3g.cu -o runs/mbar_repro_v3g -lcudart 2>&1 | grep -E "register|barriers|stack|error"
echo ""
echo "=== Verify SASS: BAR.SYNC IDs used ==="
"$CUDA/bin/cuobjdump" -sass runs/mbar_repro_v3g | grep -E 'BAR\.SYNC' | head -10

echo ""
echo "=== Run v3g — sl=300 ca=0 wnd=0 × 20 with 8s timeout each ==="
n_hang=0
n_ok=0
for i in $(seq 1 20); do
    timeout 8 runs/mbar_repro_v3g >/dev/null 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        n_ok=$((n_ok + 1))
        echo "  run $i: OK"
    else
        n_hang=$((n_hang + 1))
        echo "  run $i: HANG (rc=$rc)"
    fi
done
echo ""
echo "=== Summary: $n_ok OK / $n_hang HANG / 20 runs ==="
