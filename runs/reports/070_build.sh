#!/bin/bash
export HOME=/tmp
cd /data/lib/podman-data/projects/goml/libs
make -f Makefile.bench_r2c_e2e clean 2>&1 | tail -2
make -f Makefile.bench_r2c_e2e 2>&1 | tail -10
make -f Makefile.r2c_merged_bit_exact clean 2>&1 | tail -1
make -f Makefile.r2c_merged_bit_exact 2>&1 | tail -5
make -f Makefile.r1b_dk_bit_exact clean 2>&1 | tail -1
make -f Makefile.r1b_dk_bit_exact 2>&1 | tail -5
echo "done"
