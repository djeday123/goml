#!/bin/bash
# PC sampling v2: explicit smsp__pcsamp metrics + ncu-rep export + binary parse.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

REP="$GOML/runs/ncu_v111_pcsamp.ncu-rep"

echo "=== NCu pmsampling with explicit metrics ==="
"$NCU" \
    --target-processes all \
    --launch-skip 1 --launch-count 1 \
    --metrics "smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_wait,smsp__pcsamp_warps_issue_stalled_long_scoreboard,smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_lg_throttle,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle,smsp__pcsamp_warps_issue_stalled_mio_throttle,smsp__pcsamp_warps_issue_stalled_dispatch_stall,smsp__pcsamp_warps_issue_stalled_drain,smsp__pcsamp_warps_issue_stalled_membar,smsp__pcsamp_warps_issue_stalled_misc,smsp__pcsamp_warps_issue_stalled_no_instruction,smsp__pcsamp_warps_issue_stalled_not_selected,smsp__pcsamp_warps_issue_stalled_selected,smsp__pcsamp_warps_issue_stalled_sleeping,smsp__pcsamp_warps_issue_stalled_tex_throttle" \
    --export "$REP" \
    --import-source on \
    -f \
    "$GOML/runs/fa_v111_producer_skip" --ncu 9 2>&1 | tail -20

echo ""
echo "=== Export details via --import ==="
"$NCU" --import "$REP" --csv --print-units base --page details 2>&1 > "$GOML/runs/ncu_v111_pcsamp_details.csv"

echo "  Saved: $GOML/runs/ncu_v111_pcsamp_details.csv"
echo ""

# Now parse the per-PC stall data
python3 << 'PYEOF'
import csv, io
path = "/data/lib/podman-data/projects/goml/runs/ncu_v111_pcsamp_details.csv"
with open(path) as f: data = f.read()

# Try to find PC sampling table — typically marked by "PC", "Address", "Source File", etc.
# The format with --page details + --csv is complex.

# First let's just look for stall-related rows
print("--- Total file lines:", data.count('\n'))
print()

# Pretty-print header rows
lines = data.split('\n')
for i, l in enumerate(lines):
    if 'pcsamp' in l.lower() and i < 100:
        print(f"  Line {i}: {l[:200]}")
        break

# Count metric occurrences
from collections import Counter
counts = Counter()
for l in lines:
    if 'pcsamp' in l:
        for m in ['barrier','wait','long_scoreboard','short_scoreboard',
                  'lg_throttle','math_pipe_throttle','mio_throttle',
                  'dispatch_stall','membar','drain','no_instruction']:
            if m in l:
                counts[m] += 1
print(f"\nMetric occurrence counts:")
for m, c in counts.most_common():
    print(f"  {m:<24} {c}")
PYEOF

echo ""
echo "================================================================"
echo "Note: if details CSV doesn't have per-PC breakdown, NCu version may"
echo "limit CLI access to PC sampling table. Use 'ncu-ui' GUI on .ncu-rep file."
echo "Fallback: A/B experiment (build v112 without top __syncthreads, measure)."
echo "================================================================"
