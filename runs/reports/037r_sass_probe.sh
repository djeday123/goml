#!/bin/bash
BIN=$1
OUT_TMP=/tmp/037r_sass_$(basename "$BIN").txt
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass "$BIN" > "$OUT_TMP" 2>&1
echo "=== $BIN ==="
echo "-- mangled names with 'merged' --"
grep -oE "\b_Z[A-Za-z0-9_]*merged[A-Za-z0-9_]*\b" "$OUT_TMP" | sort -u | head -5
echo "-- STG count / kernel_merged --"
awk '/Function\s*:\s*.*merged/{f=1; print NR": "$0; next} /Function\s*:/{f=0} f' "$OUT_TMP" | head -3
echo "-- global STG.E stats --"
grep -oE "STG\.[A-Za-z0-9.]+" "$OUT_TMP" | sort | uniq -c
echo "-- global STS stats --"
grep -oE "STS\.[A-Za-z0-9.]+" "$OUT_TMP" | sort | uniq -c
echo "-- LDS stats --"
grep -oE "LDS\.[A-Za-z0-9.]+" "$OUT_TMP" | sort | uniq -c | tail -20
echo "-- total function count --"
grep -cE "^\s*Function" "$OUT_TMP"
