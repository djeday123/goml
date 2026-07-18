#!/bin/bash
# Sanity: 1 causal run to confirm ~22.2 ms baseline before NCu skew probe.
export CAUSAL=1
export CUDA_MODULE_LOADING=LAZY
cd /data/lib/podman-data/projects/goml
libs/bench_r2c_e2e
