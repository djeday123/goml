#!/bin/bash
NCU=/usr/local/cuda-13.1/bin/ncu
REP=/data/lib/podman-data/projects/goml/runs/ncu_v96b_baseline.ncu-rep
OUT=/data/lib/podman-data/projects/goml/runs/v96b_pcsamp_source.txt
"$NCU" --import "$REP" --page source --print-source sass > "$OUT" 2>&1
wc -l "$OUT"
