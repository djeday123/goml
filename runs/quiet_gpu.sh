#!/bin/bash
# 058: quiet_gpu.sh для Vugar — kill наших pattern, чужие только печать.
# ОБЯЗАТЕЛЬНО запускать до wall-замеров. НЕ убивает чужие процессы (W-ветка).

OURS_RE='r2c_merged_wall|bench_r2c_e2e|r2c_merged_bit_exact|ldmatrix_.*_probe|bench_bh1_sl8192'

echo "=== quiet_gpu.sh: GPU tenant inventory ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1

echo ""
echo "=== Наши процессы (kill) ==="
for TOOL in $(echo "$OURS_RE" | tr '|' ' '); do
    for PID in $(pgrep -f "$TOOL" 2>/dev/null); do
        NAME=$(ps -o cmd= -p "$PID" 2>/dev/null)
        ELAPSED=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
        echo "kill PID=$PID elapsed=${ELAPSED}s cmd=$NAME"
        kill -TERM "$PID" 2>&1
    done
done
sleep 3
# SIGKILL any survivors
for TOOL in $(echo "$OURS_RE" | tr '|' ' '); do
    for PID in $(pgrep -f "$TOOL" 2>/dev/null); do
        echo "SIGKILL survivor PID=$PID"
        kill -KILL "$PID" 2>&1
    done
done

echo ""
echo "=== Чужие процессы на GPU (только печать) ==="
FOREIGN=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | grep -vE "$OURS_RE")
if [ -n "$FOREIGN" ]; then
    echo "$FOREIGN"
    echo ""
    echo "WARNING: чужие GPU-процессы активны. Остановить вручную (Vugar): systemctl stop <service> или kill <PID>."
else
    echo "нет чужих процессов ✓"
fi

echo ""
echo "=== GPU idle state ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,clocks.gr,clocks.mem --format=csv
