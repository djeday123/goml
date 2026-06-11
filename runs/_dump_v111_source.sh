#!/bin/bash
/usr/local/cuda-13.1/bin/ncu --import /data/lib/podman-data/projects/goml/runs/ncu_v111_pcsampling.ncu-rep --page source --print-source sass > /data/lib/podman-data/projects/goml/runs/v111_pcsamp_source.txt 2>&1
ls -la /data/lib/podman-data/projects/goml/runs/v111_pcsamp_source.txt
