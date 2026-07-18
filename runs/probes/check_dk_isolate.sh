#!/bin/bash
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass /data/lib/podman-data/projects/goml/libs/r1b_dk_wall > /data/lib/podman-data/projects/goml/runs/probes/dk_sass_pack.txt

TXT=/data/lib/podman-data/projects/goml/runs/probes/dk_sass_pack.txt

echo "=== Function boundaries ==="
grep -n "Function :" "$TXT"
echo ""
echo "=== Wc lines ==="
wc -l "$TXT"
