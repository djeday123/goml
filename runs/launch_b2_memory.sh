#!/bin/bash
# Detached B2-BATTLE peak-memory launcher (short, ~1 min).
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/goml/runs
TS=$(date +%s)
LOG="$RUNS/b2_memory_${TS}.log"
setsid nohup go test -run TestB2_Battle_PeakMemory -timeout 15m -v ./internal/abjexam/ \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/b2_memory.pid
echo "$LOG" > /tmp/b2_memory.log_path
sleep 2
echo "PID: $PID"
echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then
    echo "ALIVE: YES"
else
    echo "ALIVE: NO"
fi
