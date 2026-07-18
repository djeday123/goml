#!/bin/bash
export HOME=/tmp
cd /data/lib/podman-data/projects/goml/libs
make -f Makefile.x1_probe_071 clean 2>&1 | tail -2
make -f Makefile.x1_probe_071 2>&1 | tail -20
