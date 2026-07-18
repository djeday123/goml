#!/usr/bin/env python3
"""AB1 RE-CERT: 30-run same-thermal for e2e_nc + e2e_c + dq_iso (sealed AA1).
   Fingerprint-gated: EXPECT kernel_dq=161 (fp16-acc), kernel_dk=248, kernel_dv_mma_p1=129.
   dK/dV не менялись — их iso пропускаем (см. V3 30-run cert).
"""
import subprocess, re, statistics as stats, math, json, os, sys

N_RUNS = 30
LIBS    = "/data/lib/podman-data/projects/goml/libs"
RESULTS = "/data/lib/podman-data/projects/goml/runs"

EXPECT = {
    "kernel_dq":         161,       # AA1 sealed (was 196)
    "kernel_dk":         248,
    "kernel_dv_mma_p1":  129,
}

CONFIGS = [
    ("ab1_e2e_nc", [f"{LIBS}/bench_e2e", "128","8192","0","0","5","20"], "weighted"),
    ("ab1_e2e_c",  [f"{LIBS}/bench_e2e", "128","8192","1","0","5","20"], "weighted"),
    ("ab1_dq_iso", [f"{LIBS}/bench_dq",  "128","8192","0","0","5","20"], "iso"),
]

def check_fingerprint(out):
    for line in out.splitlines():
        m = re.search(r"FINGERPRINT (\S+):\s*numRegs=(\d+)", line)
        if not m:
            continue
        name, regs = m.group(1), int(m.group(2))
        if name in EXPECT and regs != EXPECT[name]:
            return f"MISMATCH {name}: got {regs}, expected {EXPECT[name]}"
    return None

def parse(out, kind):
    if kind == "weighted":
        m = re.search(r"weighted TFLOPS\s*=\s*([0-9.]+)", out)
    else:
        m = re.search(r"tflops=([0-9.]+)", out)
    return float(m.group(1)) if m else None

def rawpath(name):   return f"{RESULTS}/{name}_raw.json"
def finalpath(name): return f"{RESULTS}/{name}.json"

def load_partial(name):
    p = rawpath(name)
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return []

def save_partial(name, vals):
    with open(rawpath(name), "w") as f:
        json.dump(vals, f)

def collect(name, cmd, kind):
    if os.path.exists(finalpath(name)):
        print(f"[DONE] {name} — final exists, skipping", flush=True)
        with open(finalpath(name)) as f: return json.load(f)
    vals = load_partial(name)
    already = len(vals)
    if already >= N_RUNS:
        vals = vals[:N_RUNS]
    else:
        print(f"\n=== {name} (30 runs, resuming from {already}/30) ===", flush=True)
        for i in range(already, N_RUNS):
            out = subprocess.check_output(cmd, text=True, timeout=120)
            err = check_fingerprint(out)
            if err:
                print(f"  ABORT at run {i+1}: fingerprint {err}", flush=True)
                sys.exit(2)
            v = parse(out, kind)
            vals.append(v)
            save_partial(name, vals)
            print(f"  {i+1:2d}/{N_RUNS}: {v:.3f}", flush=True)
    med = stats.median(vals); mn = stats.mean(vals); sd = stats.stdev(vals)
    cv = sd / mn * 100.0
    outliers = [(i+1, v) for i, v in enumerate(vals) if abs(v - med) > 3*sd]
    result = dict(raw=vals, median=med, mean=mn, sd=sd, cv=cv,
                  outliers=outliers, n=N_RUNS)
    with open(finalpath(name), "w") as f: json.dump(result, f, indent=2)
    return result

def main():
    print("AB1 RE-CERT — 30-run per config, fingerprint-gated", flush=True)
    print(f"Expected regs: {EXPECT}", flush=True)
    for name, cmd, kind in CONFIGS:
        r = collect(name, cmd, kind)
        print(f"[{name}] median={r['median']:.2f}  mean={r['mean']:.2f}  "
              f"sd={r['sd']:.3f}  cv={r['cv']:.3f}%  n={r['n']}  "
              f"outliers={r['outliers']}", flush=True)

if __name__ == "__main__":
    main()
