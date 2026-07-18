#!/bin/bash
# 069A ABBA: 8 pairs — dk isolated nc, E2E nc, E2E causal control.
# Strategy: build both BASE (pre-069A) and CAND (069A) binaries with distinct names,
# then run ABBA alternation without rebuilds.

export HOME=/tmp
export CUDA_MODULE_LOADING=LAZY
cd /data/lib/podman-data/projects/goml/libs

LOG=/data/lib/podman-data/projects/goml/runs/reports/069A_abba.txt
> "$LOG"

# --- STEP 1: current state is CAND (069A applied). Build CAND binaries with 069A suffix ---
echo "=== STEP 1: build CAND binaries (069A applied) ===" | tee -a "$LOG"
make -f Makefile.bench_r2c_e2e clean >/dev/null 2>&1
make -f Makefile.bench_r2c_e2e 2>&1 | tail -3 | tee -a "$LOG"
cp bench_r2c_e2e bench_r2c_e2e_069A

make -f Makefile.r1b_dk_wall clean >/dev/null 2>&1
make -f Makefile.r1b_dk_wall 2>&1 | tail -3 | tee -a "$LOG"
cp r1b_dk_wall r1b_dk_wall_069A

# --- STEP 2: roll back to BASE, build BASE binaries ---
echo "=== STEP 2: rolling back source + build BASE binaries (pre-069A) ===" | tee -a "$LOG"
cp fa_bwd_dk_new.cu.pre_069A fa_bwd_dk_new.cu
# Restore EXPECT 124 in bench (line with kernel_dk_new)
sed -i "s|kernel_dk_new,         96|kernel_dk_new,        124|" bench_r2c_e2e.cu

make -f Makefile.bench_r2c_e2e clean >/dev/null 2>&1
make -f Makefile.bench_r2c_e2e 2>&1 | tail -3 | tee -a "$LOG"
cp bench_r2c_e2e bench_r2c_e2e_BASE

make -f Makefile.r1b_dk_wall clean >/dev/null 2>&1
make -f Makefile.r1b_dk_wall 2>&1 | tail -3 | tee -a "$LOG"
cp r1b_dk_wall r1b_dk_wall_BASE

# --- STEP 3: ABBA dk isolated 8 pairs = 16 shots (BAABBABA order alternating; use ABBAABBAABBA... = 8 A + 8 B) ---
echo "" >> "$LOG"
echo "=== STEP 3: ABBA dk_new isolated wall (8 pairs = 16 shots) ===" | tee -a "$LOG"
echo "Order: B A B A B A B A A B A B A B A B (BAB × 4, then ABA × 4 — reverses to average thermal drift)" | tee -a "$LOG"
ORDER="BASE CAND BASE CAND BASE CAND BASE CAND CAND BASE CAND BASE CAND BASE CAND BASE"
i=0
for arm in $ORDER; do
    i=$((i+1))
    bin="r1b_dk_wall_${arm}"
    out=$("./$bin" 2>&1 | grep -E "median|wall" | head -1)
    echo "shot=$i arm=$arm  $out" | tee -a "$LOG"
done

# --- STEP 4: E2E nc 5-shot each arm (control on nc-verdict) ---
echo "" >> "$LOG"
echo "=== STEP 4: E2E nc (5 shots per arm) ===" | tee -a "$LOG"
for arm in BASE CAND BASE CAND BASE; do
    bin="bench_r2c_e2e_${arm}"
    out=$("./$bin" 2>&1 | grep "SEQUENTIAL" -A1 | tail -1)
    echo "arm=$arm  $out" | tee -a "$LOG"
done

# --- STEP 5: E2E causal 5-shot each arm (perенос meas) ---
echo "" >> "$LOG"
echo "=== STEP 5: E2E causal control (5 shots per arm) ===" | tee -a "$LOG"
for arm in BASE CAND BASE CAND BASE; do
    bin="bench_r2c_e2e_${arm}"
    out=$(CAUSAL=1 "./$bin" 2>&1 | grep "SEQUENTIAL" -A1 | tail -1)
    echo "arm=$arm  $out" | tee -a "$LOG"
done

# --- STEP 6: restore CAND state (069A patch back for final commit or rollback) ---
echo "" >> "$LOG"
echo "=== STEP 6: restore CAND source state (069A applied) for verdict step ===" | tee -a "$LOG"
# Re-apply 069A patch to source (was already done via cp .pre_069A to it in step 2, need to swap back)
# Currently source is pre-069A. To re-apply 069A: it's the current file state was patched, but step 2 rolled back.
# The 069A patch was: added __launch_bounds__(FA_DKN_THREADS, 5) to kernel_dk_new signature.
# Re-apply via python-safe sed or edit directly. Simplest: keep source as-is (pre-069A rolled back),
# report will decide KEEP/ROLL. If KEEP: user re-applies patch. Left rolled back for safety.
echo "Source currently rolled back to sealed (pre-069A). Verdict will decide re-apply or keep-rolled." | tee -a "$LOG"

echo "done" | tee -a "$LOG"
