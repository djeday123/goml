#!/usr/bin/env python3
"""J-SEAL: 30-run same-thermal, 5 configs, K2-sealed build.
Per-RUN checkpoint (survives any interruption). Resume from any point."""

import subprocess, re, statistics as stats, math, json, os, sys, time

N_RUNS = 30
LIBS = "/data/lib/podman-data/projects/goml/libs"
RESULTS_DIR = "/data/lib/podman-data/projects/goml/runs"

CONFIGS = [
    ("e2e_nc",  [f"{LIBS}/bench_e2e", "128", "8192", "0", "0", "5", "20"], 176.85, "weighted"),
    ("e2e_c",   [f"{LIBS}/bench_e2e", "128", "8192", "1", "0", "5", "20"], 174.0,  "weighted"),
    ("dq_iso",  [f"{LIBS}/bench_dq",  "128", "8192", "0", "0", "5", "20"], 171.9,  "iso"),
    ("dk_iso",  [f"{LIBS}/bench_dk",  "128", "8192", "0", "0", "5", "20"], 196.1,  "iso"),
    ("dv_iso",  [f"{LIBS}/bench_dv",  "128", "8192", "0", "0", "5", "20"], 160.8,  "iso"),
]

def parse(out, kind):
    if kind == "weighted":
        m = re.search(r"weighted TFLOPS\s*=\s*([0-9.]+)", out)
    else:
        m = re.search(r"tflops=([0-9.]+)", out)
    return float(m.group(1)) if m else None

def rawpath(name): return f"{RESULTS_DIR}/j_seal_{name}_raw.json"
def finalpath(name): return f"{RESULTS_DIR}/j_seal_{name}.json"

def load_partial(name):
    p = rawpath(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []

def save_partial(name, vals):
    with open(rawpath(name), "w") as f:
        json.dump(vals, f)

def collect_one(name, cmd, baseline, kind):
    if os.path.exists(finalpath(name)):
        print(f"[DONE] {name} — final exists, skipping", flush=True)
        with open(finalpath(name)) as f: return json.load(f)
    vals = load_partial(name)
    already = len(vals)
    if already >= N_RUNS:
        vals = vals[:N_RUNS]
    else:
        print(f"\n=== {name} (30 runs, baseline={baseline} T, resuming from {already}/30) ===", flush=True)
        for i in range(already, N_RUNS):
            out = subprocess.check_output(cmd, text=True, timeout=120)
            v = parse(out, kind)
            vals.append(v)
            save_partial(name, vals)   # persist EVERY run
            print(f"  {i+1:2d}/{N_RUNS}: {v:.3f}", flush=True)
    med = stats.median(vals); mn = stats.mean(vals); sd = stats.stdev(vals)
    cv = sd / mn * 100.0
    outliers = [(i+1, v) for i, v in enumerate(vals) if abs(v - med) > 3*sd]
    se = sd / math.sqrt(N_RUNS) * math.sqrt(2)
    t = (mn - baseline) / se if se > 0 else float('inf')
    delta = mn - baseline; pct = delta / baseline * 100.0
    print(f"  → median={med:.2f} T  mean={mn:.2f} T  sd={sd:.3f}  CV={cv:.3f}%", flush=True)
    if outliers: print(f"  → OUTLIERS >3σ: {outliers}", flush=True)
    print(f"  Δ vs {baseline}: {delta:+.2f} T ({pct:+.1f}%), Welch t≈{t:.2f}", flush=True)
    result = {"raw": vals, "median": med, "mean": mn, "sd": sd, "cv": cv,
              "delta": delta, "pct": pct, "t": t, "outliers": outliers,
              "baseline": baseline}
    with open(finalpath(name), "w") as f: json.dump(result, f, indent=2)
    return result

print(f"J-SEAL start {time.strftime('%H:%M:%S')}", flush=True)
results = {}
for name, cmd, baseline, kind in CONFIGS:
    results[name] = collect_one(name, cmd, baseline, kind)

print("\n" + "="*80)
print("=== FINAL J-SEAL TABLE ===")
print(f"{'Config':<12}{'baseline':>10}{'median':>10}{'mean':>10}{'sd':>8}{'CV %':>8}{'Δ %':>8}{'Welch t':>10}{'outl':>6}")
for name, r in results.items():
    print(f"{name:<12}{r['baseline']:>10.2f}{r['median']:>10.2f}{r['mean']:>10.2f}{r['sd']:>8.2f}{r['cv']:>8.3f}{r['pct']:>+8.1f}{r['t']:>+10.2f}{len(r['outliers']):>6}")
print(f"\nJ-SEAL end {time.strftime('%H:%M:%S')}", flush=True)
