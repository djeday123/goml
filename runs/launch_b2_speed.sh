#!/bin/bash
# Detached B2-BATTLE speed launcher (FA-класс: silence gate + clocks + 30-run median).
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/goml/runs
TS=$(date +%s)
LOG="$RUNS/b2_speed_${TS}.log"
setsid nohup go test -run TestB2_Battle_Speed -timeout 60m -v ./internal/abjexam/ \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/b2_speed.pid
echo "$LOG" > /tmp/b2_speed.log_path
sleep 2
echo "PID: $PID"
echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then
    echo "ALIVE: YES"
else
    echo "ALIVE: NO"
fi
