#!/bin/bash
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck /data/lib/podman-data/projects/goml/libs/bench_r2c_e2e bitexact 2>&1 | tail -15
