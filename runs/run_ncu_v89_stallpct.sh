#!/bin/bash
# NCu stall composition for v89 (P-in-regs).
# Confirm mechanism: short_scoreboard, barrier, mio_throttle should all drop.
# wait likely stays ~28% (inherent).
set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v89_pinregs_fp8"

cd "$GOML"

if [ ! -x "$NCU" ]; then echo "ERROR: ncu not at $NCU" >&2; exit 1; fi
if [ ! -x "$BIN" ]; then echo "ERROR: binary not at $BIN — run run_fa_v89_pinregs.sh first" >&2; exit 1; fi

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
    local label="$1" cfg="$2" lb="$3"
    local out="$GOML/runs/ncu_v89_stallpct_cfg${cfg}_lb${lb}.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg LB=$lb"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        "$BIN" --ncu "$cfg" "$lb" > "$out" 2>&1
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
    print("  ERROR: header not found")
    raise SystemExit(0)
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
print(f"  {'TOTAL':<24} {total:>7.2f}%")
print(f"  {'Eligible':<24} {100-total:>7.2f}%")
PYEOF
    echo ""
}

profile_cfg "v89 WIN LB=3 (413T, +3.8% vs v87)" 6 3
profile_cfg "v89 WIN LB=2 (322T, +5% vs v87)"  6 2
profile_cfg "v89 LOSS LB=3 (275T, +1.6% vs v87)" 4 3
profile_cfg "v89 LOSS LB=2 (283T, +3% vs v87)"   4 2

echo "================================================================"
echo "v87 baseline for diff (WIN LB=3):"
echo "  wait 28.93%, math_pipe 9.53%, short_scb 7.48%, mio 5.80%,"
echo "  barrier 3.57%, no_instr 2.56%, long_scb 2.44%, TOTAL 66.37%"
echo ""
echo "v89 expected (P-in-regs attacks 3 stalls):"
echo "  short_scoreboard ↓ (smP load removed)"
echo "  barrier ↓ (2 syncs removed)"
echo "  mio_throttle ↓ (smP stores removed)"
echo "  wait ≈ same 28% (inherent)"
echo "  math_pipe ≈ same 10% (architectural)"
echo "================================================================"
