#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/runs/mbar_repro_v3j
echo "=== v3j ×20 (timeout 30s/run) ==="
n_ok=0
n_hang=0
for i in $(seq 1 20); do
    timeout 30 "$BIN" > /dev/null 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        n_ok=$((n_ok + 1))
        echo "  run $i: OK (rc=0)"
    else
        n_hang=$((n_hang + 1))
        echo "  run $i: HANG/FAIL (rc=$rc)"
    fi
done
echo ""
echo "=== Summary: $n_ok OK / $n_hang HANG / 20 runs ==="
