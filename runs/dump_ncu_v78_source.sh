#!/bin/bash
# Dump source-attributed metrics from v78 .ncu-rep.
# Goal: identify where PMS samples actually pile up post-V-overlap to confirm
# smP STS+sync is the next biggest stall site (vs other surprises).
set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
RPT=/data/lib/podman-data/projects/goml/runs/ncu_v78_full.ncu-rep
OUT_SRC=/data/lib/podman-data/projects/goml/runs/ncu_v78_source.csv
OUT_DET=/data/lib/podman-data/projects/goml/runs/ncu_v78_details.csv

"$NCU" --import "$RPT" --page source --csv --print-units base > "$OUT_SRC" 2>&1 || true
echo "Source page: $(wc -l <"$OUT_SRC") lines → $OUT_SRC"

"$NCU" --import "$RPT" --page details --csv --print-units base > "$OUT_DET" 2>&1 || true
echo "Details page: $(wc -l <"$OUT_DET") lines → $OUT_DET"

echo ""
echo "=== Top stall reasons (Warp State Statistics rule) ==="
grep -iE "stall_|warp_state|CPIStall|Stall" "$OUT_DET" | head -25

echo ""
echo "=== Top SASS instructions by sample count ==="
echo "(Column 3 = total samples, Column 4 = stall samples, sorted by stall samples)"
echo ""
# Find SASS lines (start with hex address). Column 4 is "stall samples".
# Sort numerically descending by column 4.
grep -E '^"0x[0-9a-f]+","      ' "$OUT_SRC" | \
    awk -F'","' '{
        sass = $2
        stall = $4
        total = $3
        gsub(/^"/,"",total); gsub(/"$/,"",total)
        printf "%6d stall  %6d total  %s\n", stall+0, total+0, sass
    }' | sort -rn | head -25

echo ""
echo "=== STS / smP / smV_T attributed stores by stall samples ==="
grep -E '^"0x[0-9a-f]+","      STS' "$OUT_SRC" | \
    awk -F'","' '{
        sass = $2; stall = $4; total = $3
        printf "%6d stall  %6d total  %s\n", stall+0, total+0, sass
    }' | sort -rn | head -15

echo ""
echo "=== LDS attributed loads by stall samples ==="
grep -E '^"0x[0-9a-f]+","      LDS' "$OUT_SRC" | \
    awk -F'","' '{
        sass = $2; stall = $4; total = $3
        printf "%6d stall  %6d total  %s\n", stall+0, total+0, sass
    }' | sort -rn | head -15

echo ""
echo "=== HMMA / QMMA (MMA) instructions by stall samples ==="
grep -E '^"0x[0-9a-f]+","      (HMMA|QMMA|MMA)' "$OUT_SRC" | \
    awk -F'","' '{
        sass = $2; stall = $4; total = $3
        printf "%6d stall  %6d total  %s\n", stall+0, total+0, sass
    }' | sort -rn | head -15

echo ""
echo "=== FFMA / MUFU (softmax math) by stall samples ==="
grep -E '^"0x[0-9a-f]+","      (FFMA|MUFU|FADD|FMUL)' "$OUT_SRC" | \
    awk -F'","' '{
        sass = $2; stall = $4; total = $3
        printf "%6d stall  %6d total  %s\n", stall+0, total+0, sass
    }' | sort -rn | head -15

echo ""
echo "=== BAR.SYNC (__syncthreads) attributed by stall ==="
grep -E '^"0x[0-9a-f]+","      (BAR|MEMBAR)' "$OUT_SRC" | \
    awk -F'","' '{
        sass = $2; stall = $4; total = $3
        printf "%6d stall  %6d total  %s\n", stall+0, total+0, sass
    }' | sort -rn | head -10
