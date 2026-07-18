#!/bin/bash
set -uo pipefail
GOML=/data/lib/podman-data/projects/goml
BASE=$GOML/runs/fa_v121r_diet
NEW=$GOML/runs/fa_v121r_sp1

# 30 alternating runs, single config (bh=64 sl=8192 wnd=0): главная мишень.
for run in $(seq 1 30); do
    "$BASE" --time 64 8192 0 5 2>&1 | grep TFLOPS | awk -v r="$run" -v src="v121r" '{print src","r","$5}'
    "$NEW"  --time 64 8192 0 5 2>&1 | grep TFLOPS | awk -v r="$run" -v src="sp1"   '{print src","r","$5}'
done | tee /dev/stderr | python3 -c '
import sys, statistics
data = {"v121r":[], "sp1":[]}
for line in sys.stdin:
    parts = line.strip().split(",")
    if len(parts) < 3: continue
    src = parts[0]
    val = parts[2].replace("TFLOPS=", "")
    try: data[src].append(float(val))
    except: pass
v121r = data["v121r"]
sp1 = data["sp1"]
m1 = statistics.mean(v121r)
m2 = statistics.mean(sp1)
s1 = statistics.stdev(v121r)
s2 = statistics.stdev(sp1)
n = min(len(v121r), len(sp1))
print(f"\n=== 30-run summary bh=64 sl=8192 wnd=0 (TFLOPS) ===")
print(f"v121r : mean={m1:.2f}  sd={s1:.2f}  min={min(v121r):.2f}  max={max(v121r):.2f}  n={len(v121r)}")
print(f"sp1   : mean={m2:.2f}  sd={s2:.2f}  min={min(sp1):.2f}  max={max(sp1):.2f}  n={len(sp1)}")
delta = m2 - m1
sigma_diff = ((s1*s1 + s2*s2)/n)**0.5
print(f"\nΔmean = {delta:+.2f}T ({100.0*delta/m1:+.2f}%)")
print(f"σ_diff = {sigma_diff:.2f}T  (Welch t)")
print(f"|Δ|/σ_diff = {abs(delta)/sigma_diff:.2f}")
if abs(delta) > 2*sigma_diff:
    print(f"VERDICT: ACCEPT (|Δ| > 2σ)")
else:
    print(f"VERDICT: NOISE (|Δ| ≤ 2σ — v121r остаётся)")
'
