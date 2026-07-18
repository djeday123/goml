#!/bin/bash
BIN=/data/lib/podman-data/projects/goml/libs/r1b_dk_wall
OUT=/data/lib/podman-data/projects/goml/runs/probes/dk_w2_full_sass.txt
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" > "$OUT" 2>&1
START=$(grep -n "Function : _ZN13fa_bwd_dk_new" "$OUT" | head -1 | cut -d: -f1)
END=$(grep -n "^\s*\.section\|Function : _ZN9fa_bwd_dk9kernel_dk" "$OUT" | awk -F: -v s=$START '$1>s{print $1;exit}')
[ -z "$END" ] && END=$(wc -l < "$OUT")
sed -n "${START},${END}p" "$OUT" > /tmp/dk_new_full.txt
echo "=== SASS boundaries ==="
echo "Start: $START, End: $END, Lines: $((END - START + 1))"
echo ""
echo "=== Full kernel_dk_new SASS counters ==="
echo -n "LDL/STL: "; grep -cE "\bLDL\b|\bSTL\b" /tmp/dk_new_full.txt
echo -n "SHFL: "; grep -c "SHFL" /tmp/dk_new_full.txt
echo -n "LDS (any): "; grep -cE "^\s+/\*[0-9a-f]+\*/\s+LDS" /tmp/dk_new_full.txt
echo -n "LDS.32: "; grep -cE "LDS\.32\b" /tmp/dk_new_full.txt
echo -n "LDS.U8/U16: "; grep -cE "LDS\.(U8|U16|B8|B16)" /tmp/dk_new_full.txt
echo -n "STS (any): "; grep -cE "^\s+/\*[0-9a-f]+\*/\s+STS" /tmp/dk_new_full.txt
echo -n "STS.32: "; grep -cE "STS\.32\b" /tmp/dk_new_full.txt
echo -n "STS.U8/U16: "; grep -cE "STS\.(U8|U16|B8|B16)" /tmp/dk_new_full.txt
echo -n "PRMT: "; grep -c "PRMT" /tmp/dk_new_full.txt
echo -n "SEL: "; grep -c "SEL R" /tmp/dk_new_full.txt
echo -n "BAR.SYNC: "; grep -c "BAR" /tmp/dk_new_full.txt
echo -n "LDGSTS (cp.async): "; grep -c "LDGSTS" /tmp/dk_new_full.txt
echo -n "QMMA (MMA): "; grep -c "QMMA" /tmp/dk_new_full.txt
