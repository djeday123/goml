#!/bin/bash
# Dump source-attributed metrics from ncu .ncu-rep.
set -euo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
RPT=/data/lib/podman-data/projects/goml/runs/ncu_v69_full.ncu-rep
OUT_SRC=/data/lib/podman-data/projects/goml/runs/ncu_v69_source.csv
OUT_DET=/data/lib/podman-data/projects/goml/runs/ncu_v69_details.csv
OUT_RAW=/data/lib/podman-data/projects/goml/runs/ncu_v69_raw.csv

# Source-attributed page (per source line)
"$NCU" --import "$RPT" --page source --csv --print-units base > "$OUT_SRC" 2>&1 || true
echo "Source page: $(wc -l <"$OUT_SRC") lines → $OUT_SRC"

# Details page (per-section summary)
"$NCU" --import "$RPT" --page details --csv --print-units base > "$OUT_DET" 2>&1 || true
echo "Details page: $(wc -l <"$OUT_DET") lines → $OUT_DET"

# Raw page (all metric values)
"$NCU" --import "$RPT" --page raw --csv --print-units base > "$OUT_RAW" 2>&1 || true
echo "Raw page: $(wc -l <"$OUT_RAW") lines → $OUT_RAW"

echo ""
echo "=== Top stall reasons (Warp State Statistics) ==="
grep -iE "stall_|warp.*stall|Stall" "$OUT_DET" | head -30

echo ""
echo "=== Top SMEM bank-conflict source lines (from source CSV, if attributed) ==="
grep -iE "bank.*conflict|shared.*store|smem.*store|l1tex.*shared" "$OUT_SRC" | head -30

echo ""
echo "=== Pipeline utilization (FMA/ALU/Tensor/LSU/FP64/XU) ==="
grep -iE "pipe.*util|fma|alu|tensor|lsu|xu " "$OUT_DET" | grep -iE "%|elapsed_active" | head -30
