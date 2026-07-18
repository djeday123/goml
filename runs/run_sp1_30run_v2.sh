#!/bin/bash
set -uo pipefail
BASE=/data/lib/podman-data/projects/goml/runs/fa_v121r_diet
NEW=/data/lib/podman-data/projects/goml/runs/fa_v121r_sp1

for run in $(seq 1 30); do
    OUT_BASE=$("$BASE" --time 64 8192 0 5 2>&1 | grep "median=")
    OUT_NEW=$("$NEW"  --time 64 8192 0 5 2>&1 | grep "median=")
    MED_BASE=$(echo "$OUT_BASE" | grep -oE 'median=[0-9.]+' | tr -d 'median=')
    MED_NEW=$(echo "$OUT_NEW"   | grep -oE 'median=[0-9.]+' | tr -d 'median=')
    TFLOPS_BASE=$(echo "$OUT_BASE" | grep -oE 'TFLOPS=[0-9.]+' | tr -d 'TFLOPS=')
    TFLOPS_NEW=$(echo "$OUT_NEW"   | grep -oE 'TFLOPS=[0-9.]+' | tr -d 'TFLOPS=')
    echo "v121r,$run,$MED_BASE,$TFLOPS_BASE"
    echo "sp1,$run,$MED_NEW,$TFLOPS_NEW"
done | tee /tmp/sp1_log.csv | python3 -c '
import sys, statistics
data = {"v121r":[], "sp1":[]}
for line in sys.stdin:
    parts = line.strip().split(",")
    if len(parts) < 4: continue
    src = parts[0]
    try: data[src].append(float(parts[3]))   # TFLOPS column
    except: pass
v = data["v121r"]; s = data["sp1"]
m1, m2 = statistics.mean(v), statistics.mean(s)
sd1, sd2 = statistics.stdev(v), statistics.stdev(s)
n = min(len(v), len(s))
print(f"\n=== 30-run bh=64 sl=8192 wnd=0 ===")
print(f"v121r TFLOPS: mean={m1:.2f} sd={sd1:.2f} min={min(v):.2f} max={max(v):.2f} n={len(v)}")
print(f"sp1   TFLOPS: mean={m2:.2f} sd={sd2:.2f} min={min(s):.2f} max={max(s):.2f} n={len(s)}")
delta = m2 - m1
sig = ((sd1**2 + sd2**2)/n)**0.5
print(f"Δmean = {delta:+.2f}T ({100*delta/m1:+.2f}%)")
print(f"σ_diff = {sig:.2f}T,  |Δ|/σ = {abs(delta)/sig:.2f}")
if abs(delta) > 2*sig:
    print(f"VERDICT: SIGNIFICANT (|Δ|>2σ): SP1 {\"ACCEPT\" if delta>0 else \"REJECT\"}")
else:
    print(f"VERDICT: NOISE")
'
