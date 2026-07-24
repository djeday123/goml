#!/bin/bash
# Detached B2-BATTLE gotorch/v6 regression from goml/runs launcher.
cd /data/lib/podman-data/projects/gotorch/v6
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/gotorch/v6/runs
TS=$(date +%s)
LOG="$RUNS/b2_gotorch_regression_${TS}.log"
setsid nohup bash -c 'go test -timeout 30m -short ./... 2>&1; echo "=== B2_RUN_COMPLETE ==="' \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/b2_gotorch_reg.pid
echo "$LOG" > /tmp/b2_gotorch_reg.log_path
sleep 2
echo "PID: $PID"
echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then
    echo "ALIVE: YES"
else
    echo "ALIVE: NO"
fi
