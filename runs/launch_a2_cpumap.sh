#!/bin/bash
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/goml/runs
TS=$(date +%s)
LOG="$RUNS/a2_cpumap_${TS}.log"
setsid nohup go test -run 'TestA2_Battle_CPUMap' -timeout 15m -v ./internal/abjexam/ \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/a2_cpumap.pid
sleep 2
echo "PID: $PID"; echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then echo "ALIVE: YES"; else echo "ALIVE: NO"; fi
