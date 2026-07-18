#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
"$NCU" --import /tmp/dk_source.ncu-rep --page source --print-source per-inst 2>&1 | head -40
echo "==== TRY: --details ===="
"$NCU" --import /tmp/dk_source.ncu-rep --page source 2>&1 | head -30
