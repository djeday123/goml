#!/bin/bash
export INJECT_BITFLIP=1
/data/lib/podman-data/projects/goml/libs/r2c_merged_bit_exact 2>&1 | tail -8
echo "Exit: $?"
