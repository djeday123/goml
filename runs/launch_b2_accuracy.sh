#!/bin/bash
# Detached B2-BATTLE accuracy launcher — feedback-detached-long-runs pattern.
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/goml/runs
TS=$(date +%s)
LOG="$RUNS/b2_accuracy_${TS}.log"
setsid nohup go test -run TestB2_Battle_Accuracy -timeout 60m -v ./internal/abjexam/ \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/b2_accuracy.pid
echo "$LOG" > /tmp/b2_accuracy.log_path
sleep 2
echo "PID: $PID"
echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then
    echo "ALIVE: YES"
else
    echo "ALIVE: NO"
fi
