#!/usr/bin/env python3
"""
Benchmark: Tri Dao FA2 vs GoML v20 on identical configs.
Corrected FLOPS = 4·s²·d·h / 2 for causal (per Tri Dao FA2 paper).
"""

import torch
import ctypes
import os
import subprocess
import sys
import json

def run_single_config(batch, nheads, seqlen, headdim, label):
    """Run one config in isolated subprocess to avoid CUDA context corruption."""
    code = f'''
import torch
import ctypes
import os

torch.manual_seed(42)
device = torch.device("cuda:0")

# Build GoML .so
so_path = "libs/flash_attention_v20.so"
if not os.path.exists(so_path):
    os.system("nvcc -O3 -arch=sm_89 -std=c++17 --shared -Xcompiler -fPIC libs/flash_attention_v20.cu -o " + so_path)

goml_lib = ctypes.CDLL(so_path)
goml_lib.flash_attention_v20_forward.restype = ctypes.c_int
goml_lib.flash_attention_v20_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]

from flash_attn import flash_attn_func

B, H, S, D = {batch}, {nheads}, {seqlen}, {headdim}
total_heads = B * H
flops = total_heads * (4.0 * S * S * D) / 2.0
iters = 100 if S <= 1024 else (20 if S <= 4096 else 10)

def bench(fn, warmup=5, iters=iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# Allocate as (total_heads, S, D) — guaranteed flat layout for GoML
Q_flat = torch.randn(total_heads, S, D, device=device, dtype=torch.float16)
K_flat = torch.randn(total_heads, S, D, device=device, dtype=torch.float16)
V_flat = torch.randn(total_heads, S, D, device=device, dtype=torch.float16)
O_flat = torch.zeros_like(Q_flat)

# GoML benchmark
def run_goml():
    ret = goml_lib.flash_attention_v20_forward(
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), O_flat.data_ptr(),
        total_heads, S, D, 1, ctypes.c_void_p(0))
    return ret

# Quick sanity check
ret = run_goml()
torch.cuda.synchronize()
if ret != 0:
    print("GOML_ERR")
    exit(1)

ms_goml = bench(run_goml)
t_goml = flops / (ms_goml / 1000.0) / 1e12

# FA2 benchmark — needs (B, S, H, D)
Q_fa2 = Q_flat.reshape(B, H, S, D).transpose(1, 2).contiguous()
K_fa2 = K_flat.reshape(B, H, S, D).transpose(1, 2).contiguous()
V_fa2 = V_flat.reshape(B, H, S, D).transpose(1, 2).contiguous()

ms_fa2 = bench(lambda: flash_attn_func(Q_fa2, K_fa2, V_fa2, causal=True))
t_fa2 = flops / (ms_fa2 / 1000.0) / 1e12

import json
print("RESULT:" + json.dumps({{"fa2": t_fa2, "goml": t_goml}}))
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120
    )

    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])

    if "GOML_ERR" in result.stdout:
        return {"error": "GoML kernel error"}
    if result.returncode != 0:
        # Extract short error
        err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
        return {"error": err[:80]}

    return {"error": "no result"}


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"=== FA2 vs GoML v20 — Head-to-Head ===")
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")

    GPU_PEAK = 165.2

    configs = [
        # Issue #477: hidden=2048, headdim=128, nheads=16, total_tokens=16384
        ("issue-128s",  128, 16, 128,  128),
        ("issue-256s",   64, 16, 256,  128),
        ("issue-512s",   32, 16, 512,  128),
        ("issue-1024s",  16, 16, 1024, 128),
        # 7B: 32 heads
        ("7B-512s",      1, 32, 512,  128),
        ("7B-1024s",     1, 32, 1024, 128),
        ("7B-2048s",     1, 32, 2048, 128),
        ("7B-4096s",     1, 32, 4096, 128),
        # 70B: 64 heads
        ("70B-512s",     1, 64, 512,  128),
        ("70B-2048s",    1, 64, 2048, 128),
    ]

    print(f"\ncausal=True, FLOPS = 4·s²·d·h / 2")
    print(f"Each config runs in isolated subprocess\n")

    hdr = f"{'Config':<14} {'B':>4} {'H':>4} {'S':>5}  {'FA2':>9} {'GoML':>9} {'GoML/FA2':>8} {'FA2%':>6} {'GoML%':>6}"
    print(hdr)
    print("-" * len(hdr))

    for label, batch, nheads, seqlen, headdim in configs:
        r = run_single_config(batch, nheads, seqlen, headdim, label)

        if "error" in r:
            print(f"{label:<14} {batch:>4} {nheads:>4} {seqlen:>5}  ERROR: {r['error']}")
            continue

        t_fa2 = r["fa2"]
        t_goml = r["goml"]
        ratio = t_goml / t_fa2 if t_fa2 > 0 else 0

        print(f"{label:<14} {batch:>4} {nheads:>4} {seqlen:>5}  {t_fa2:>7.2f} T {t_goml:>7.2f} T {ratio:>7.2f}x {t_fa2/GPU_PEAK*100:>5.1f}% {t_goml/GPU_PEAK*100:>5.1f}%")

    print(f"\nPeak = {GPU_PEAK} TFLOPS (FP16 HMMA, FP32 accum, RTX 4090)")

if __name__ == "__main__":
    main()
