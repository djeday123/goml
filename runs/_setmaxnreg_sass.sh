#!/bin/bash
CUDA=/usr/local/cuda-13.1
"$CUDA/bin/cuobjdump" -sass /data/lib/podman-data/projects/goml/runs/setmaxnreg_probe | head -40
