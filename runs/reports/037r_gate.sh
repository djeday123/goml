#!/bin/bash
# 058: gate с проверкой тишины GPU (наследие 057-STOP)
# EXPECT merged fingerprint + чужие compute-apps -> ABORT

BIN=/data/lib/podman-data/projects/goml/libs/r2c_merged_wall
EXPECT=252  # 057 stand restored: prod = 040 sealed

# === Замок стенда (058-a): чужие compute-apps на GPU ===
# Наши имена — r2c_merged_wall, bench_r2c_e2e, r2c_merged_bit_exact
OURS_RE='r2c_merged_wall|bench_r2c_e2e|r2c_merged_bit_exact|ldmatrix_.*_probe|ncu|compute-sanitizer'
FOREIGN=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE "$OURS_RE" | grep -v "^$")
if [ -n "$FOREIGN" ]; then
    echo "GATE ABORT: foreign GPU compute-apps present:" >&2
    echo "$FOREIGN" >&2
    echo "Run runs/quiet_gpu.sh или остановить чужие процессы вручную." >&2
    exit 2
fi

# === Замок стенда: наш зомби (elapsed > 5 min = подозрение) ===
OUR_PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -E "$OURS_RE" | awk -F',' '{print $1}' | tr -d ' ')
for PID in $OUR_PIDS; do
    ELAPSED=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
    if [ -n "$ELAPSED" ] && [ "$ELAPSED" -gt 300 ]; then
        NAME=$(ps -o comm= -p "$PID" 2>/dev/null)
        echo "GATE WARN: own long-running process PID=$PID ($NAME) elapsed=${ELAPSED}s. Kill вручную если это зомби." >&2
    fi
done

# === Fingerprint gate ===
LINE=$("$BIN" 2>&1 | grep FINGERPRINT | head -1)
echo "$LINE"
REGS=$(echo "$LINE" | grep -oE "numRegs=[0-9]+" | head -1 | grep -oE "[0-9]+")
if [ "$REGS" != "$EXPECT" ]; then
    echo "GATE ABORT: numRegs=$REGS != EXPECT=$EXPECT" >&2
    exit 1
fi
echo "GATE OK: numRegs=$REGS matches EXPECT=$EXPECT + GPU-quiet (no foreign compute-apps)"
