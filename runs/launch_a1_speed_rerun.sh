#!/bin/bash
# A-3.0 forensic: re-run A-1 speed test fresh session, verify 19.4 ms reproduces.
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/goml/runs
TS=$(date +%s)
LOG="$RUNS/a1_speed_rerun_${TS}.log"
setsid nohup go test -run 'TestA1_Battle_Speed' -timeout 30m -v ./internal/abjexam/ \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/a1_rerun.pid
sleep 2
echo "PID: $PID"; echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then echo "ALIVE: YES"; else echo "ALIVE: NO"; fi
