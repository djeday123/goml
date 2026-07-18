#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/r1b_dk_wall
OUT=/data/lib/podman-data/projects/goml/runs/probes/dk_w2_sass.txt
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" > "$OUT" 2>&1
# Find kernel_dk_new body
grep -n "Function : _ZN13fa_bwd_dk_new" "$OUT" | head -1
END=$(grep -n "Function : _ZN9fa_bwd_dk9kernel_dk" "$OUT" | head -1 | cut -d: -f1)
START=$(grep -n "Function : _ZN13fa_bwd_dk_new" "$OUT" | head -1 | cut -d: -f1)
if [ -z "$END" ]; then END=$(wc -l < "$OUT"); fi
sed -n "${START},${END}p" "$OUT" > /tmp/dk_new_only.txt
echo "=== kernel_dk_new SASS gates ==="
echo -n "LDL/STL: "; grep -cE "\bLDL\b|\bSTL\b" /tmp/dk_new_only.txt
echo -n "SHFL: "; grep -c SHFL /tmp/dk_new_only.txt
echo -n "STS.U8/16: "; grep -cE "STS\.(U8|U16|B8|B16)" /tmp/dk_new_only.txt
echo -n "STS.32: "; grep -cE "^\s+/\*[0-9a-f]+\*/\s+STS\b\.32|^\s+/\*[0-9a-f]+\*/\s+STS\b\s" /tmp/dk_new_only.txt
