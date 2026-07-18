#!/usr/bin/env python3
import statistics, subprocess, os
BASE = "/data/lib/podman-data/projects/goml/runs/fa_v121r_diet"
NEW  = "/data/lib/podman-data/projects/goml/runs/fa_v121r_sp1"
def run(bin):
    out = subprocess.run([bin, "--time", "64", "8192", "0", "5"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "TFLOPS=" in line:
            for tok in line.split():
                if tok.startswith("TFLOPS="):
                    return float(tok.split("=")[1])
    return None
v=[]; s=[]
for i in range(30):
    a = run(BASE); b = run(NEW)
    v.append(a); s.append(b)
    print(f"run {i+1:2d}: v121r={a:.2f}  sp1={b:.2f}", flush=True)
m1, m2 = statistics.mean(v), statistics.mean(s)
sd1, sd2 = statistics.stdev(v), statistics.stdev(s)
n = len(v)
delta = m2 - m1
sig = ((sd1**2 + sd2**2)/n)**0.5
verdict = "NOISE"
if abs(delta) > 2*sig:
    verdict = "SIGNIFICANT: SP1 ACCEPT" if delta > 0 else "SIGNIFICANT: SP1 REJECT"
print()
print("=== 30-run bh=64 sl=8192 wnd=0 ===")
print(f"v121r TFLOPS: mean={m1:.2f} sd={sd1:.2f} min={min(v):.2f} max={max(v):.2f} n={n}")
print(f"sp1   TFLOPS: mean={m2:.2f} sd={sd2:.2f} min={min(s):.2f} max={max(s):.2f} n={n}")
print(f"Delta = {delta:+.2f}T ({100*delta/m1:+.2f}%)")
print(f"sigma_diff = {sig:.2f}T, |Delta|/sigma = {abs(delta)/sig:.2f}")
print(f"VERDICT: {verdict}")
