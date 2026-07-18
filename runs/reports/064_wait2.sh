#!/bin/bash
FILE=/data/lib/podman-data/projects/goml/runs/reports/064_clone_30run_causal.txt
until [ "$(grep -c '^run=' "$FILE" 2>/dev/null || echo 0)" -ge 30 ]; do
    sleep 10
done
echo "done: $(grep -c '^run=' "$FILE") runs"
