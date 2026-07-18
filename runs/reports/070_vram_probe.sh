#!/bin/bash
# 070 VRAM probe: run bench_r2c_e2e in background with longer iters, poll nvidia-smi.
# Compare peak memory: current (070 applied) vs pre-070 archive.
export HOME=/tmp
export CUDA_MODULE_LOADING=LAZY
cd /data/lib/podman-data/projects/goml/libs

LOG=/data/lib/podman-data/projects/goml/runs/reports/070_vram_probe.txt
> "$LOG"

# Get pre-alloc baseline
BASE_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "pre-bench baseline: ${BASE_MB} MB" | tee -a "$LOG"

# --- 070 CURRENT ---
echo "--- CURRENT (070 applied, dS_T = nullptr) ---" | tee -a "$LOG"
./bench_r2c_e2e >/dev/null 2>&1 &
BPID=$!
sleep 0.5
CURR_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "peak during bench: ${CURR_MB} MB (delta = $((CURR_MB - BASE_MB)) MB above baseline)" | tee -a "$LOG"
wait $BPID

# Wait for GPU to release
sleep 1

# --- ROLLBACK to pre-070 source, rebuild ---
echo "" | tee -a "$LOG"
echo "--- Rebuilding with pre-070 source for BASE comparison ---" | tee -a "$LOG"
cp bench_r2c_e2e.cu.pre_070 bench_r2c_e2e.cu
make -f Makefile.bench_r2c_e2e clean >/dev/null 2>&1
make -f Makefile.bench_r2c_e2e 2>&1 | tail -3 | tee -a "$LOG"

echo "--- BASE (pre-070, dS_T allocated 8.59 GB) ---" | tee -a "$LOG"
./bench_r2c_e2e >/dev/null 2>&1 &
BPID=$!
sleep 0.5
BASE_PEAK_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "peak during bench: ${BASE_PEAK_MB} MB (delta = $((BASE_PEAK_MB - BASE_MB)) MB above baseline)" | tee -a "$LOG"
wait $BPID

# Wait
sleep 1

# --- Reapply 070 patch (restore final state) ---
echo "" | tee -a "$LOG"
echo "--- Reapplying 070 patch (final state = CURRENT) ---" | tee -a "$LOG"
# Use two sed to add "; dS_T = nullptr" to both alloc sites
sed -i 's|CKR(cudaMalloc(\&dS_nat,dsz));CKR(cudaMalloc(\&dS_T,dsz));|CKR(cudaMalloc(\&dS_nat,dsz)); dS_T = nullptr;   // 070: dS_T dead-alloc removed|' bench_r2c_e2e.cu

make -f Makefile.bench_r2c_e2e clean >/dev/null 2>&1
make -f Makefile.bench_r2c_e2e 2>&1 | tail -3 | tee -a "$LOG"

# Verify final state
grep -n "dS_T = nullptr" bench_r2c_e2e.cu | tee -a "$LOG"

# --- Summary ---
DELTA=$((BASE_PEAK_MB - CURR_MB))
echo "" | tee -a "$LOG"
echo "===== VRAM DELTA =====" | tee -a "$LOG"
echo "BASE peak: ${BASE_PEAK_MB} MB" | tee -a "$LOG"
echo "070  peak: ${CURR_MB} MB" | tee -a "$LOG"
echo "delta: -${DELTA} MB = ~$(echo "scale=2; $DELTA/1024" | bc) GB" | tee -a "$LOG"
echo "expected: -8590 MB (~8.59 GB)" | tee -a "$LOG"
echo "done" | tee -a "$LOG"
