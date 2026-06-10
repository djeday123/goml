#!/bin/bash
# Source-attributed NCu on v81 WIN LB=3 (bh=16 sl=4096, +27% perf, No Eligible 43%).
# Find the EXACT source line(s) driving the 43% No Eligible — not aggregate, per-PC.
# Then choose: ILP softmax (if softmax line dominates), recompute (if reg-pressure
# source), or some other lever, based on data not guess.
set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v81_hd64_fp8"
RPT="$GOML/runs/ncu_v81_win_lb3.ncu-rep"
SRC="$GOML/runs/ncu_v81_win_lb3_source.csv"
DET="$GOML/runs/ncu_v81_win_lb3_details.csv"

cd "$GOML"

if [ ! -x "$NCU" ]; then echo "ERROR: ncu not at $NCU" >&2; exit 1; fi
if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not found — build v81 first" >&2; exit 1; fi

echo "=== Standalone sanity (no profiler) ==="
"$BIN" --ncu 6 3 2>&1 | head -3
echo ""

# Step 1: profile WIN LB=3 with pmsampling, save .ncu-rep.
# pmsampling collects per-PC stall samples — needed for source attribution.
echo "=== Step 1: NCu pmsampling capture → $RPT ==="
"$NCU" \
    --target-processes all \
    --launch-skip 1 --launch-count 1 \
    --set pmsampling \
    --import-source on \
    --export "$RPT" -f \
    "$BIN" --ncu 6 3 2>&1 | tail -10
echo "ncu exit: $?"
echo ""

# Step 2: dump raw per-PC samples (this is where pmsampling stalls live).
RAW="$GOML/runs/ncu_v81_win_lb3_raw.csv"
echo "=== Step 2: extract raw per-PC samples → $RAW ==="
"$NCU" --import "$RPT" --page raw --csv --print-units base > "$RAW" 2>&1
echo "wrote $(wc -l < "$RAW") lines"

echo "=== Step 3: extract details page → $DET ==="
"$NCU" --import "$RPT" --page details --csv --print-units base > "$DET" 2>&1
echo "wrote $(wc -l < "$DET") lines"

# Step 4: human-readable source-annotated dump (text, not CSV).
SRCTXT="$GOML/runs/ncu_v81_win_lb3_source.txt"
echo "=== Step 4: source-annotated text view → $SRCTXT ==="
"$NCU" --import "$RPT" --page source --print-source on > "$SRCTXT" 2>&1
echo "wrote $(wc -l < "$SRCTXT") lines"

# Step 5: also try src CSV (the SASS dump, kept as before).
echo "=== Step 5: SASS source CSV → $SRC ==="
"$NCU" --import "$RPT" --page source --csv --print-units base > "$SRC" 2>&1
echo "wrote $(wc -l < "$SRC") lines"
echo ""

# Step 6: scan raw CSV columns
echo "=== RAW CSV column headers (first 30 cols) ==="
head -1 "$RAW" 2>/dev/null | tr ',' '\n' | head -30
echo ""

echo "=== Stall-related columns in RAW CSV ==="
head -1 "$RAW" 2>/dev/null | tr ',' '\n' | grep -nE "(stall|pcsamp|sample|eligible)" -i | head -20
echo ""

echo "=== KERNEL-AGGREGATE stall ranking (sorted by sample count) ==="
# Extract pmsampling stall metric values from RAW CSV — kernel-level aggregate.
python3 << PYEOF
import csv
with open("$RAW") as f:
    rdr = csv.reader(f)
    rows = list(rdr)
if len(rows) < 2:
    print("Need >=2 rows in raw CSV")
else:
    header = rows[0]
    data = rows[1]
    stalls = []
    for i, col in enumerate(header):
        if 'pmsampling' in col.lower() and 'stalled' in col.lower():
            try:
                val = float(data[i]) if data[i] not in ('','""') else 0.0
            except ValueError:
                val = 0.0
            short = col.replace('pmsampling:smsp__warps_issue_stalled_','').rstrip('"').rstrip('.avg')
            stalls.append((short, val))
    stalls.sort(key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in stalls)
    print(f"{'Stall reason':<28} {'Value':>10} {'%':>7}")
    print('-'*48)
    for name, val in stalls:
        pct = (val/total*100) if total else 0
        print(f"{name:<28} {val:>10.4f} {pct:>6.2f}%")
    print('-'*48)
    print(f"{'TOTAL':<28} {total:>10.4f}")
PYEOF
echo ""

echo "=== Per-PC pcsamp ranking (if any non-zero) ==="
python3 << PYEOF
import csv
with open("$RAW") as f:
    rdr = csv.reader(f)
    rows = list(rdr)
if len(rows) < 2:
    print("Need >=2 rows")
else:
    header = rows[0]
    data = rows[1]
    pcsamp = []
    for i, col in enumerate(header):
        if 'pcsamp' in col.lower() and 'stalled' in col.lower():
            try:
                val = float(data[i]) if data[i] not in ('','""') else 0.0
            except ValueError:
                val = 0.0
            short = col.replace('warpsampling:smsp__pcsamp_warps_issue_stalled_','').rstrip('"')
            pcsamp.append((short, val))
    pcsamp.sort(key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in pcsamp if v>0)
    if total > 0:
        print(f"{'PC-sampled stall':<28} {'Samples':>10} {'%':>7}")
        for name, val in pcsamp:
            if val > 0:
                pct = (val/total*100) if total else 0
                print(f"{name:<28} {val:>10.4f} {pct:>6.2f}%")
    else:
        print("(no non-zero pcsamp values in RAW — per-PC may need different export)")
PYEOF
echo ""

echo "=== Text source-annotated view (rows with .cu attribution) ==="
grep -nE "flash_attention_v81|\.cu:[0-9]+" "$SRCTXT" 2>/dev/null | head -30
echo ""

echo "=== Source lines with non-zero sample counts (from $SRCTXT) ==="
# NCu text source view shows: source_line | sample_count | sass_lines
# Filter rows where there's a number column populated.
awk '/flash_attention_v81|\.cu:[0-9]/ {print}' "$SRCTXT" 2>/dev/null | head -40
echo ""

echo "================================================================"
echo "Artifacts:"
echo "  Profile:        $RPT"
echo "  Raw per-PC CSV: $RAW"
echo "  Details CSV:    $DET"
echo "  SASS CSV:       $SRC"
echo "  Source text:    $SRCTXT"
echo "Browse: ncu-ui $RPT"
echo "================================================================"
