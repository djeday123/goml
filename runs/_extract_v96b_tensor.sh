#!/bin/bash
NCU=/usr/local/cuda-13.1/bin/ncu
REP=/data/lib/podman-data/projects/goml/runs/ncu_v96b_baseline.ncu-rep
"$NCU" --import "$REP" --page details 2>&1 | grep -iE "tensor|pipe|qmma|util" | head -30
echo "---"
"$NCU" --import "$REP" --csv --page raw 2>&1 | grep -iE "tensor|pipe_tensor" | head -10
