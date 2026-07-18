#!/bin/bash
# 037-r: SASS структурный маркер pre-T-cut vs post-T-cut
OUT=/data/lib/podman-data/projects/goml/runs/reports/037r_sass_wall.txt
BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
> "$OUT"
echo "=== SASS marker check r2c_merged_wall (Jul 6 07:14) ===" | tee -a "$OUT"
echo "kernel_merged_v1 mangled name search" | tee -a "$OUT"
/usr/local/cuda-13.1/bin/cuobjdump --function _ZN16fa_bwd_merged_v116kernel_merged_v1EPKhS1_S1_PK6__halfPKfS6_PhS7_Pfiiiiif --dump-sass "$BIN" 2>&1 > /tmp/sass_wall.txt
cp /tmp/sass_wall.txt "$OUT.raw"
echo "" | tee -a "$OUT"
echo "STG.E.SYS.128 count (drain dS_nat + dS_T если pre-cut):" | tee -a "$OUT"
grep -c "STG.E.128\|STG.E.SYS.128\|STG.E.SYS" /tmp/sass_wall.txt | tee -a "$OUT"
echo "STG.E.b64 count:" | tee -a "$OUT"
grep -c "STG.E.b64" /tmp/sass_wall.txt | tee -a "$OUT"
echo "STS instr counts:" | tee -a "$OUT"
grep -cE "STS\.32|STS\.16|STS\.U8|STS\.b16|STS\.128" /tmp/sass_wall.txt | tee -a "$OUT"
echo "Individual STG types:" | tee -a "$OUT"
grep -oE "STG\.[A-Za-z0-9.]+" /tmp/sass_wall.txt | sort | uniq -c | tee -a "$OUT"
