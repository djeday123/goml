#!/bin/bash
# NCu stall composition for v98 (K-preload on hd=128 v96 base).
# Compare to v96 baseline to see if short_scoreboard 6.50% actually dropped.
# Key signal: if short_scb at peak dropped by 0.5-1.5pp, mechanism worked despite
# bench saying null/marginal. If short_scb unchanged, K-preload had zero effect.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v98_kpreload"

cd "$GOML"

if [ ! -x "$NCU" ]; then echo "ERROR: ncu not at $NCU" >&2; exit 1; fi
if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not built" >&2; exit 1; fi

METRICS=$(cat <<EOF | tr '\n' ',' | sed 's/,$//'
smsp__warp_issue_stalled_wait_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct
smsp__warp_issue_stalled_drain_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
smsp__warp_issue_stalled_misc_per_warp_active.pct
smsp__warp_issue_stalled_sleeping_per_warp_active.pct
smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct
smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct
EOF
)

profile_cfg() {
    local label="$1" cfg="$2"
    local out="$GOML/runs/ncu_v98_stallpct_cfg${cfg}.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$BIN" --ncu "$cfg" > "$out" 2>&1
    echo "wrote $out"
    python3 << PYEOF
import csv, io
with open("$out") as f:
    lines = f.readlines()
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'):
        hi = i; break
if hi is None:
    print("  ERROR: header not found"); raise SystemExit(0)
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
m = []
for r in rdr:
    n = r.get('Metric Name','')
    if 'stalled' in n.lower() and 'per_warp_active' in n.lower():
        try: v = float(r.get('Metric Value','0').replace(',','.'))
        except: v = 0
        short = n.replace('smsp__warp_issue_stalled_','').replace('_per_warp_active.pct','')
        m.append((short, v))
m.sort(key=lambda x: x[1], reverse=True)
total = sum(v for _,v in m)
print(f"  {'Stall reason':<24} {'%':>8}")
for n,v in m:
    print(f"  {n:<24} {v:>7.2f}%")
print(f"  {'TOTAL stalled':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

profile_cfg "SMALL  bh=8  sl=2048 (220T)"  2
profile_cfg "MID    bh=16 sl=4096 (417T)"  6
profile_cfg "LARGE  bh=64 sl=8192 (568T) ← PEAK"  9

echo "================================================================"
echo "REFERENCE — v96 hd=128 PEAK (bh=64 sl=8192, 568.5T):"
echo "  wait               37.77%"
echo "  math_pipe_throttle  8.86%"
echo "  short_scoreboard    6.50%  ← target of K-preload"
echo "  mio_throttle        3.91%"
echo "  lg_throttle         3.31%"
echo "  dispatch_stall      2.92%"
echo "  barrier             2.00%"
echo "  TOTAL              69.54% → Eligible 30.46%"
echo ""
echo "WATCH FOR (peak cfg=9):"
echo "  short_scb 6.50 → 5.0-6.0%  → K-preload mechanism worked"
echo "  short_scb 6.50 → 6.5%      → null, K-preload did nothing"
echo "  wait 37.77 → 37.8% ± 0.2pp → wait unchanged (inherent)"
echo "  math_pipe 8.86 → 9.5-10%   → slightly more MMA pressure (preload moved work)"
echo "  Eligible 30.46 → 30.7-31% → tiny gain consistent with bench +0.16% median"
echo "================================================================"
