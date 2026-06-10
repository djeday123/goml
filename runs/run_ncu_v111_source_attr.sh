#!/bin/bash
# Source-attributed NCu for v111 PEAK. Identify WHICH instruction (which __syncthreads,
# bar.sync, cpa_wait) dominates barrier and wait stalls.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

# Source-counter section: includes per-PC stall sampling.
# --import-source on: keeps source mapping
# --target-processes all: profile all child processes
OUT="$GOML/runs/ncu_v111_source.csv"

echo "=== Source-attributed NCu profile for v111 PEAK cfg=9 ==="
"$NCU" \
    --target-processes all \
    --launch-skip 1 --launch-count 1 \
    --set source \
    --import-source on \
    --csv \
    "$GOML/runs/fa_v111_producer_skip" --ncu 9 > "$OUT" 2>&1

echo "  Raw output: $OUT"
echo ""
echo "=== Top stall locations (by source line) ==="

python3 << 'PYEOF'
import csv, io
with open("/data/lib/podman-data/projects/goml/runs/ncu_v111_source.csv") as f:
    lines = f.readlines()

# Find header row
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'):
        hi = i; break

if hi is None:
    # Try other header
    for i, l in enumerate(lines[:200]):
        if 'Metric Name' in l and ',' in l:
            print(f"  Found possible header at line {i}: {l[:200]}")

if hi is None:
    print("  No standard CSV header — dumping first 100 lines for diagnosis:")
    for i, l in enumerate(lines[:100]):
        print(f"  {i}: {l.rstrip()}")
    raise SystemExit(0)

rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))

# Group by (file, line) and stall metric
per_line = {}  # (file, line) -> {metric: value}
for r in rdr:
    n = r.get('Metric Name', '')
    src = r.get('Source File', '') or r.get('File', '')
    ln = r.get('Source Line', '') or r.get('Line', '')
    if not src or not ln:
        continue
    if 'stall' not in n.lower():
        continue
    try:
        v = float(r.get('Metric Value', '0').replace(',', '.'))
    except:
        v = 0
    key = (src, ln)
    if key not in per_line:
        per_line[key] = {}
    per_line[key][n] = v

# Top barrier
barrier_metrics = [m for m in {m for d in per_line.values() for m in d.keys()} if 'barrier' in m.lower()]
wait_metrics    = [m for m in {m for d in per_line.values() for m in d.keys()} if 'wait' in m.lower() and 'lg' not in m.lower()]

def top_by_metric(metric_name, label, top_n=10):
    rows = []
    for (src, ln), d in per_line.items():
        if metric_name in d and d[metric_name] > 0:
            rows.append((d[metric_name], src, ln))
    rows.sort(reverse=True)
    print(f"\n--- TOP {top_n} stall by {label} ({metric_name}) ---")
    for v, src, ln in rows[:top_n]:
        print(f"  {v:8.2f}  {src}:{ln}")

print("\nAvailable barrier metrics:", barrier_metrics[:5])
print("Available wait metrics:",    wait_metrics[:5])

for m in barrier_metrics:
    top_by_metric(m, "BARRIER")

for m in wait_metrics[:2]:
    top_by_metric(m, "WAIT")
PYEOF

echo ""
echo "================================================================"
echo "After this we know which specific sync/wait dominates."
echo "  If barrier mostly at top-of-iter __syncthreads → producer-skip top sync"
echo "  If wait mostly at top cpa_wait<0> → producer's own cp.async waiting"
echo "================================================================"
