#!/bin/bash
# NCu stall composition for v96 hd=128 — never measured before.
# Hypothesis: hd=128 may have DIFFERENT dominant stalls than hd=64 (v89: wait
# 28%, math_pipe 11%, short_scb 6.6%, mio 4.5%). hd=128 has 2× MMAs per kv-iter
# and SMEM-bound 2 blocks/SM (vs hd=64's 3) — different bottleneck possible.
#
# Profile 3 regime points:
#   cfg=2 bh=8 sl=2048 (small wave-tail)
#   cfg=6 bh=16 sl=4096 (mid — was +4% from ks-batched)
#   cfg=9 bh=64 sl=8192 (large PEAK — production champion 568T)

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v96_ksbatched"

cd "$GOML"

# Rebuild v96 to include new --ncu mode
echo "=== Rebuild v96 with --ncu CLI ==="
/usr/local/cuda-13.1/bin/nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    -Xptxas=-v -lineinfo \
    libs/flash_attention_v96_ksbatched_hd128_fp8_forward.cu \
    -o runs/fa_v96_ksbatched -lcudart 2>&1 | grep -E "(register|spill|error)" | head -5

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
    local out="$GOML/runs/ncu_v96_stallpct_cfg${cfg}.csv"
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
echo "REFERENCE — v89 hd=64 WIN LB=3 (413T) stall composition:"
echo "  wait                28.40%  (TOP, inherent)"
echo "  math_pipe_throttle  10.61%"
echo "  short_scoreboard     6.63%"
echo "  mio_throttle         4.48%"
echo "  barrier              3.48%"
echo "  TOTAL stalled       64.27% → Eligible 35.73%"
echo ""
echo "WATCH FOR: any v96 stall > 15% that's not wait → potential new lever"
echo "================================================================"
