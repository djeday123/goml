#!/bin/bash
# Detached B2-BATTLE regression: goml (excluding long abjexam) + FA-canary.
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/goml/runs
TS=$(date +%s)
LOG="$RUNS/b2_goml_regression_${TS}.log"
# Excluded: internal/abjexam (12min accuracy test we already ran).
setsid nohup bash -c 'go test -timeout 30m -short ./... 2>&1; echo "=== B2_RUN_COMPLETE ==="' \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/b2_goml_reg.pid
echo "$LOG" > /tmp/b2_goml_reg.log_path
sleep 2
echo "PID: $PID"
echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then
    echo "ALIVE: YES"
else
    echo "ALIVE: NO"
fi
