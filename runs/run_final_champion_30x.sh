#!/bin/bash
# ============================================================
# OFFICIAL 30-RUN MEASUREMENT OF FORWARD FA CHAMPION (v121r)
# Pure measurement protocol — no comparisons, no interpretation.
# ============================================================
set -uo pipefail

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
SRC=$GOML/libs/flash_attention_v121r_diet_hd128_fp8_forward.cu
BIN=$GOML/runs/fa_v121r_diet

# --- Preamble: thermal/clock/driver snapshot ---
echo "==================================================================="
echo "FORWARD FA CHAMPION OFFICIAL MEASUREMENT"
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "Host: $(hostname)"
echo "==================================================================="
echo ""
echo "--- GPU snapshot (thermal/clock start) ---"
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw \
           --format=csv
echo ""

# --- Binary check ---
if [ ! -x "$BIN" ]; then
    echo "Binary $BIN NOT FOUND — rebuilding (one-time; ptxas summary will be printed)"
    "$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
        -Xptxas=-v -lineinfo "$SRC" -o "$BIN" -lcudart 2>&1 \
        | grep -E "register|spill|stack|warning|error" | head -10
else
    echo "--- Using existing binary $BIN (NO rebuild) ---"
    ls -la "$BIN"
    echo ""
    echo "--- ptxas summary (extracted from rebuild test, side-channel for record) ---"
    "$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
        -Xptxas=-v "$SRC" -o /tmp/_v121r_ptxas_check -lcudart 2>&1 \
        | grep -E "register|spill|stack" | head -5
    rm -f /tmp/_v121r_ptxas_check
fi
echo ""

measure_form() {
    local BH=$1 SL=$2 LABEL=$3
    echo "==================================================================="
    echo "FORM: $LABEL  (bh=$BH sl=$SL wnd=0, grid=$((BH * (SL + 127) / 128)))"
    echo "==================================================================="
    echo ""
    echo "--- Warmup: 10 launches (no recording, clock stabilization) ---"
    for i in 1 2 3 4 5 6 7 8 9 10; do
        "$BIN" --time "$BH" "$SL" 0 5 > /dev/null 2>&1
    done
    echo "warmup done"
    echo ""

    echo "--- Main measurement: 30 runs (each = median-of-5 launches) ---"
    local LOG_FILE
    LOG_FILE=$(mktemp /tmp/champion_${BH}x${SL}_XXXXXX.log)
    for run in $(seq 1 30); do
        OUT=$("$BIN" --time "$BH" "$SL" 0 5 2>&1 | grep TFLOPS)
        TFLOPS=$(echo "$OUT" | grep -oE 'TFLOPS=[0-9.]+' | head -1 | tr -d 'TFLOPS=')
        SMS=$(nvidia-smi --query-gpu=clocks.current.sm,temperature.gpu \
              --format=csv,noheader,nounits | head -1)
        CLOCK=$(echo "$SMS" | awk -F, '{print $1}' | tr -d ' ')
        TEMP=$(echo "$SMS" | awk -F, '{print $2}' | tr -d ' ')
        printf "  run %2d: %7.2f T  (sm_clock %s MHz, temp %s C)\n" \
               "$run" "$TFLOPS" "$CLOCK" "$TEMP"
        echo "$TFLOPS" >> "$LOG_FILE"
    done
    echo ""

    echo "--- Statistics ---"
    python3 - "$LOG_FILE" "$LABEL" <<'PYEOF'
import sys, statistics
path = sys.argv[1]
label = sys.argv[2]
vals = []
with open(path) as f:
    for line in f:
        try: vals.append(float(line.strip()))
        except: pass
n = len(vals)
mean = statistics.mean(vals)
sd = statistics.stdev(vals)
med = statistics.median(vals)
mn = min(vals)
mx = max(vals)
first10 = vals[:10]
last10 = vals[-10:]
m1 = statistics.mean(first10)
m2 = statistics.mean(last10)
sd1 = statistics.stdev(first10)
sd2 = statistics.stdev(last10)
drift = m2 - m1
sigma_diff = ((sd1**2 + sd2**2)/10)**0.5
print(f"  n          = {n}")
print(f"  mean       = {mean:.2f} T")
print(f"  sigma      = {sd:.2f} T")
print(f"  median     = {med:.2f} T")
print(f"  min / max  = {mn:.2f} / {mx:.2f} T  (range {mx - mn:.2f} T)")
print()
print(f"  Thermal-drift check (mean of first 10 vs last 10):")
print(f"    first 10 mean = {m1:.2f} T  (sd {sd1:.2f})")
print(f"    last 10 mean  = {m2:.2f} T  (sd {sd2:.2f})")
print(f"    drift        = {drift:+.2f} T,  sigma_diff = {sigma_diff:.2f} T,  |drift|/sigma = {abs(drift)/sigma_diff:.2f}")
if abs(drift) > 2 * sigma_diff:
    print(f"    NOTE: thermal drift detected (>2 sigma) — see sm_clock log per run")
else:
    print(f"    drift within noise (<2 sigma)")
PYEOF
    rm -f "$LOG_FILE"
    echo ""
}

measure_form 64  8192 "bh=64 sl=8192 wnd=0  (primary peak target)"
measure_form 128 8192 "bh=128 sl=8192 wnd=0 (secondary peak)"

echo "==================================================================="
echo "--- GPU snapshot (thermal/clock end) ---"
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw \
           --format=csv
echo "==================================================================="
echo "DONE. End time: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "==================================================================="
