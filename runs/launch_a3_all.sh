#!/bin/bash
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
RUNS=/data/lib/podman-data/projects/goml/runs
TS=$(date +%s)
LOG="$RUNS/a3_all_${TS}.log"
# Chain: loss (fast sanity, bit-exact) -> speed (30-run, quiet GPU) -> cpumap (10 iters, quiet GPU)
setsid nohup bash -c '
  go test -run TestA3_Battle_LossOnly -timeout 15m -v ./internal/abjexam/ 2>&1
  echo "=== A3 LossOnly done ==="
  sleep 10
  go test -run TestA3_Battle_Speed -timeout 30m -v ./internal/abjexam/ 2>&1
  echo "=== A3 Speed done ==="
  sleep 5
  go test -run TestA3_Battle_CPUMap -timeout 15m -v ./internal/abjexam/ 2>&1
  echo "=== B2_RUN_COMPLETE ==="
' > "$LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > /tmp/a3_all.pid
sleep 2
echo "PID: $PID"; echo "LOG: $LOG"
if kill -0 $PID 2>/dev/null; then echo "ALIVE: YES"; else echo "ALIVE: NO"; fi
