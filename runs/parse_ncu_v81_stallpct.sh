#!/bin/bash
# Re-parse existing CSV files from run_ncu_v81_stallpct.sh — no NCu re-run needed.
# CSV format is row-per-metric, not column-per-metric. Original parser was broken.
set -u

GOML=/data/lib/podman-data/projects/goml

parse_one() {
    local label="$1" cfg="$2" lb="$3"
    local csv="$GOML/runs/ncu_v81_stallpct_cfg${cfg}_lb${lb}.csv"

    echo "================================================================"
    echo "  $label  ($csv)"
    echo "================================================================"
    if [ ! -f "$csv" ]; then
        echo "  CSV missing — run run_ncu_v81_stallpct.sh first"
        return
    fi

    python3 << PYEOF
import csv
import io

# NCu writes stderr-style status lines BEFORE the actual CSV header.
# Skip until we find the line starting with "ID","Process ID".
with open("$csv") as f:
    lines = f.readlines()
header_idx = None
for i, line in enumerate(lines):
    if line.startswith('"ID","Process ID"'):
        header_idx = i
        break
if header_idx is None:
    print("  ERROR: could not find CSV header row")
    raise SystemExit(0)
csv_text = ''.join(lines[header_idx:])

metrics = []
rdr = csv.DictReader(io.StringIO(csv_text))
for row in rdr:
    name = row.get('Metric Name', '')
    if 'stalled' in name.lower() and 'per_warp_active' in name.lower():
        try:
            val = float(row.get('Metric Value', '0').replace(',', '.'))
        except ValueError:
            val = 0.0
        short = name.replace('smsp__warp_issue_stalled_', '').replace('_per_warp_active.pct', '')
        metrics.append((short, val))

metrics.sort(key=lambda x: x[1], reverse=True)
total = sum(v for _, v in metrics)
print(f"  {'Stall reason':<24} {'% per warp active':>20}")
print('  ' + '-'*46)
for name, val in metrics:
    print(f"  {name:<24} {val:>18.2f}%")
print('  ' + '-'*46)
print(f"  {'TOTAL stalled':<24} {total:>18.2f}%")
print(f"  {'Eligible (~100-total)':<24} {100-total:>18.2f}%")
PYEOF
    echo ""
}

parse_one "WIN  LB=3 (perf 390T, +27%)" 6 3
parse_one "WIN  LB=2 (perf 308T)"       6 2
parse_one "LOSS LB=3 (perf 261T, -8%)"  4 3
parse_one "LOSS LB=2 (perf 284T)"       4 2

echo "================================================================"
echo "Top stalls in WIN LB=3 = best target for next lever."
echo "  wait dominates? → ILP softmax / break MMA dep chain"
echo "  short_scoreboard dominates? → P-in-regs / SMEM round-trip elim"
echo "  long_scoreboard dominates? → multi-stage K async"
echo "  math_pipe_throttle dominates? → near ceiling, hard to push"
echo "================================================================"
