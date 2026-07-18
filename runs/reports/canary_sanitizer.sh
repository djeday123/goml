#!/bin/bash
export HOME=/tmp
SANITIZER=/usr/local/cuda-13.1/compute-sanitizer/compute-sanitizer
if [ ! -x "$SANITIZER" ]; then
    SANITIZER=$(which compute-sanitizer 2>/dev/null)
fi
echo "Using: $SANITIZER"
$SANITIZER --version 2>&1 | head -3
echo "---"
# Run only CANARY form by modifying stdin (not directly supported; run full but grep)
"$SANITIZER" --tool memcheck --launch-timeout 60 \
    /data/lib/podman-data/projects/goml/libs/r1b_dk_bit_exact 2>&1 | head -80
