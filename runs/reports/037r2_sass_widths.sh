#!/bin/bash
OUT_MERGED=/data/lib/podman-data/projects/goml/runs/reports/037r2_sass_merged_only.txt

echo "=== FULL LDS width taxonomy (kernel_merged_v1 fresh) ==="
echo "-- All LDS mnemonics --"
grep -oE "\bLDS(\.[A-Za-z0-9]+)*" "$OUT_MERGED" | sort | uniq -c

echo ""
echo "-- Detailed LDS with byte width (raw match) --"
grep -oE "LDS(\.E)?\.(32|64|128|U8|U16|S8|S16|B32|B64|b16|b32|b64)" "$OUT_MERGED" | sort | uniq -c

echo ""
echo "-- Sample raw LDS lines (first 30) --"
grep -E "^\s*/\*" "$OUT_MERGED" | grep -E "\bLDS\b" | head -30

echo ""
echo "-- All STS mnemonics --"
grep -oE "\bSTS(\.[A-Za-z0-9]+)*" "$OUT_MERGED" | sort | uniq -c
