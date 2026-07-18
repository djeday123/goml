#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/r1c_dq_wall
TXT=/tmp/dq_sass_pack.txt
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" > "$TXT" 2>&1
echo "=== SASS gates on kernel_dq_new ==="
awk '/Function : _ZN13fa_bwd_dq_new/,/^\.section|^\s*Function :/' "$TXT" > /tmp/dq_body.txt
echo -n "LDL/STL total: "; grep -cE "\bLDL\b|\bSTL\b" /tmp/dq_body.txt
echo -n "SHFL: "; grep -cE "SHFL" /tmp/dq_body.txt
echo -n "STS.32 (STS not U8/U16): "; grep -cE "STS\b" /tmp/dq_body.txt
echo -n "STS.U8/16: "; grep -cE "STS\.(U8|U16|B8|B16)" /tmp/dq_body.txt
