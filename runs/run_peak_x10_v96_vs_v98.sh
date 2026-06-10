#!/bin/bash
# Variance ×10 statistical comparison v96 vs v98 on PEAK config (bh=64 sl=8192).
# Print now includes best/med/worst/mean/sd + RAW: all 10 values.
# Source already modified (Edit tool) to print all values. We only need to bump
# VARIANCE_RUNS = 3 → 30 for this run, then revert.

set -uo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

V96_SRC="libs/flash_attention_v96_ksbatched_hd128_fp8_forward.cu"
V98_SRC="libs/flash_attention_v98_kpreload_hd128_fp8_forward.cu"

# Backup originals
cp "$V96_SRC" "$V96_SRC.bak"
cp "$V98_SRC" "$V98_SRC.bak"
trap 'mv "$V96_SRC.bak" "$V96_SRC"; mv "$V98_SRC.bak" "$V98_SRC"' EXIT

echo "=== Step 1: VARIANCE_RUNS = 3 → 30 ==="
sed -i 's/const int VARIANCE_RUNS = 3;/const int VARIANCE_RUNS = 30;/' "$V96_SRC"
sed -i 's/const int VARIANCE_RUNS = 3;/const int VARIANCE_RUNS = 30;/' "$V98_SRC"

echo "=== Step 2: build _x10 binaries ==="
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    "$V96_SRC" -o runs/fa_v96_x10 -lcudart 2>&1 | grep -E "(error)" | head -3
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
    "$V98_SRC" -o runs/fa_v98_x10 -lcudart 2>&1 | grep -E "(error)" | head -3

echo ""
echo "=== Step 3: run v96 (30× peak bh=64 sl=8192) ==="
V96_LINE=$(runs/fa_v96_x10 2>&1 | grep "bh=64  sl=8192")
echo "$V96_LINE"

echo ""
echo "=== Step 4: run v98 (30× peak bh=64 sl=8192) ==="
V98_LINE=$(runs/fa_v98_x10 2>&1 | grep "bh=64  sl=8192")
echo "$V98_LINE"

echo ""
echo "=== Step 5: t-test analysis ==="
python3 << PYEOF
import re, statistics
def parse(line):
    # Format:  bh=64  sl=8192 hd=128 ca=0 wnd=0     best=... med=... worst=... mean=... sd=... RAW: 567.50 ...
    m = re.search(r"RAW:(.*)$", line.strip())
    if not m: return None
    vals = [float(x) for x in m.group(1).split() if x]
    return vals

v96 = parse("""$V96_LINE""")
v98 = parse("""$V98_LINE""")

if not v96 or not v98:
    print("could not parse — check binaries built correctly")
    raise SystemExit(1)

print(f"\nv96 raw (n={len(v96)}): {v96}")
print(f"v98 raw (n={len(v98)}): {v98}")
m96 = statistics.mean(v96)
m98 = statistics.mean(v98)
sd96 = statistics.stdev(v96) if len(v96)>1 else 0
sd98 = statistics.stdev(v98) if len(v98)>1 else 0
print(f"\nv96: mean={m96:.2f} sd={sd96:.2f} min={min(v96):.2f} max={max(v96):.2f}")
print(f"v98: mean={m98:.2f} sd={sd98:.2f} min={min(v98):.2f} max={max(v98):.2f}")
delta = m98 - m96
pct = 100 * delta / m96
print(f"\nΔ mean v98−v96 = {delta:+.2f} T  ({pct:+.3f}%)")
pooled = ((sd96**2 + sd98**2)/2) ** 0.5
print(f"pooled σ       = {pooled:.2f} T")
if pooled > 0.001:
    se = pooled * (2.0/len(v96))**0.5
    t = delta / se
    print(f"Welch t-stat   = {t:+.2f}  (SE = {se:.2f})")
    if abs(t) > 2.0: verdict = "STATISTICALLY SIGNIFICANT (|t| > 2)"
    elif abs(t) > 1.0: verdict = "marginal (1 < |t| < 2)"
    else: verdict = "WITHIN NOISE (|t| < 1)"
    print(f"Verdict: {verdict}")
PYEOF
