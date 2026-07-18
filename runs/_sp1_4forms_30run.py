#!/usr/bin/env python3
import statistics, subprocess
BASE = "/data/lib/podman-data/projects/goml/runs/fa_v121r_diet"
NEW  = "/data/lib/podman-data/projects/goml/runs/fa_v121r_sp1"
FORMS = [
    (64,  8192, 0,    "bh=64 sl=8192 wnd=0   (long, main target)"),
    (32,  2048, 0,    "bh=32 sl=2048 wnd=0   (short, lower bound)"),
    (64,  8192, 1024, "bh=64 sl=8192 wnd=1024 (window, 89% hit)"),
    (4,   4096, 0,    "bh=4  sl=4096 wnd=0    (wave-tail, dispatcher)"),
]
def run(bin, bh, sl, wnd):
    out = subprocess.run([bin, "--time", str(bh), str(sl), str(wnd), "5"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "TFLOPS=" in line:
            for tok in line.split():
                if tok.startswith("TFLOPS="):
                    return float(tok.split("=")[1])
    return None

print()
print(f"{'Form':<48}  {'v121r mean':>12}  {'SP1 mean':>12}  {'delta%':>8}  {'|d|/sd':>8}  verdict")
print("-" * 110)
for bh, sl, wnd, label in FORMS:
    v=[]; s=[]
    for i in range(30):
        v.append(run(BASE, bh, sl, wnd))
        s.append(run(NEW,  bh, sl, wnd))
    m1, m2 = statistics.mean(v), statistics.mean(s)
    sd1, sd2 = statistics.stdev(v), statistics.stdev(s)
    n = len(v)
    delta = m2 - m1
    sig = ((sd1**2 + sd2**2)/n)**0.5
    d_pct = 100.0 * delta / m1
    ratio = abs(delta) / sig
    if ratio > 2 and delta > 0:
        verd = "ACCEPT"
    elif ratio > 2 and delta < 0:
        verd = "REJECT"
    else:
        verd = "noise"
    print(f"{label:<48}  {m1:>10.2f} (sd={sd1:.2f})  {m2:>10.2f} (sd={sd2:.2f})  {d_pct:>+7.2f}%  {ratio:>7.2f}  {verd}")
