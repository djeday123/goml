#!/bin/bash
FILE=/data/lib/podman-data/projects/goml/runs/reports/064_clone_30run_causal.txt
until [ "$(wc -l < "$FILE" 2>/dev/null || echo 0)" -ge 35 ]; do
    sleep 5
done
echo "done"
wc -l "$FILE"
