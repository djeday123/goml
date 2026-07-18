#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall
for k in 1 2 3 4 5; do
    "$BIN" 2>&1 | grep -E "avg_ms|FINGERPRINT" | head -2
    echo "---"
done
