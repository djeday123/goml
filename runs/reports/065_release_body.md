## FP8 FlashAttention backward — first public sm_120a implementation

Custom fused FP8 (e4m3) attention backward for **NVIDIA RTX PRO 6000 Blackwell Workstation Edition** (`sm_120a`) / consumer Blackwell. World-first FP8 backward we could find at this shape and architecture; if you have a matching public number, please open an issue — we'll add it.

### Certified numbers (30-run clean-clone verify, 064)

**Canonical form**: `bh=128, sl=8192, hd=128`, seed=42, per-run warmup=5 iters=20.
Fingerprint gate (register-count 252/124/69/38) + gate-silence (no foreign compute-apps) on every run.

| Track | Wall (30-run median) | CV | Throughput (labelled convention) |
|:--|:-:|:-:|:--|
| **Non-causal** | **42.346 ms** (cert 062) / 42.352 ms (064 clean-clone) | 0.085–0.098% | **415.44 T proj (16N²d, Tri Dao V3 = 8 MMA)** · **259.65 T fused (10N²d, R2C actual = 5 MMA)** |
| **Causal** | **22.206 ms** (cert 063) / 22.231 ms (064 clean-clone) | 0.063–0.074% | ~260 T fused per active tile (same 10N²d convention as nc); ~495 T effective if divided by causal wall (naïve, double-counts skipped triangle — reported here only as a comparison anchor, not as the throughput claim) |

**Both wall-times are counter-verified real work.** Under `CAUSAL=1` the qt-loops skip tiles with `i < kt` (standard causal-attention optimisation): NCu confirms instruction count, DRAM bytes, and shared-memory wavefronts all drop ~50 % (ratio 0.505–0.506 on the three iter-loop kernels; D=1.0 on the control kernel that has no qt-loop). `std::chrono` cross-checks the CUDA-event wall within +0.53 %.

Use whichever throughput convention matches the reference you compare against — we always label which one we're printing.

### Reproducibility (one command from a clean clone)

```bash
git clone <REPO_URL> && cd fa-blackwell-fp8 && git checkout v0.2.0 && ./verify.sh
```

`verify.sh` runs five stages: build → fingerprint 252/124/69/38 → merged bit-exact 11/11 → dK bit-exact 11/11 (incl. CANARY bh=1 sl=300 wnd=96) → 5-run wall in both TFLOPS conventions labelled. Requires CUDA 13.1+, driver 580.159.03+, `sm_120a` GPU.

Source md5 shipped in this tag (Apache-2.0 + SPDX headers baked in):

```
src/fa_bwd_dk_new.cu       eb492e0729ef643280591b8c8dd8a29d
src/fa_bwd_merged_v1.cu    720774c28807d01214adff16c9003221
src/fa_bwd_dq_new.cu       7660bd960cc39c799d588c573bb47c5d
src/fa_bwd_common.cuh      5a948c2e8005f569424f0b4e8c25928e
```

### Where this sits vs the public field (honest map)

Comparing attention kernels is a minefield of unlabelled TFLOPS conventions and mismatched shapes. Below is the best map we can draw without waving hands.

| Reference | Same shape (bh=128, sl=8192, hd=128, bwd) | Notes |
|:--|:--|:--|
| **FA2 on A100 (BF16 bwd)** | ~175 TFLOPS in the 10N²d convention (literature estimate, not our run) — we sit at ~260 T fused, ≈**×1.5** at same convention on our nc wall. Using 16N²d proj vs the same reference the ratio is ~×1.75. | Different silicon (A100 vs RTX PRO 6000 Blackwell). Same convention. Same shape family. |
| **FA3 on H100 (BF16/FP8 bwd)** | H100 has more raw silicon and lower host-side latency; FA3's bwd is faster in **absolute** ms than ours. Where we win is **per-dollar**: RTX PRO 6000 Blackwell is roughly ×2 cheaper per unit of this workload at time of writing. | Different silicon, different price point. FA3 wins absolute; we win per-dollar. |
| **FA3-style bwd on H100 at this exact shape, bwd, causal** | **empty chair at the table** | If you have H100 access, please run `benchmark_flash_attention.py` at `bh=128, sl=8192, hd=128, bwd, causal` (BF16 or FP8) and drop the number in an issue — we'll add it here with your callsign. |

We are **not** publishing a headline ×N speedup against a datacenter GPU. The fair story is per-dollar plus the "sm_120a had no public FP8 backward before this." Bare cross-silicon TFLOPS comparisons at mismatched conventions don't survive a same-convention same-shape audit, so we won't lead with one.

### Known limitations

- **Shape**: `hd = 128` only (canonical form for the release cert). Other head-dims not supported without changes to the harness and swizzle constants.
- **Architecture**: `sm_120a` only. Datacenter Blackwell (`sm_100`) and older (sm_80/89/90) not supported without rebuild + probable retuning.
- **Precision**: FP8 (e4m3) with FP16 for O/dO and FP32 accumulators. No BF16-only or FP16-only backward path.
- **Causal wall** is honest wall for actually-executed work — not a "TFLOPS = full-matrix / causal wall" number (that would double-count the skipped triangle).
- **FP64 floors** (inherent to FP8, not bugs): dK ~4.65e-3 nc / ~3.1e-2 causal; dV ~4.75e-3 nc / ~3.2e-2 causal; dQ ~4.87e-3 nc / ~3.3e-2 causal.

### License

Apache License 2.0 — every source carries `SPDX-License-Identifier: Apache-2.0`.

### Reports (this cut)

- `062_cert400.md` — cert package, 30-run non-causal.
- `063_causal_release.md` + `063r_causal_release_fix.md` — causal 30-run + NCu work-detector + chrono cross-check.
- `064_release_final.md` — Apache-2.0 + SPDX + clean-clone verify (5/5 + 30-run × 2).

Copyright (c) 2026 Vugar and the FA-Blackwell-fp8 authors.
