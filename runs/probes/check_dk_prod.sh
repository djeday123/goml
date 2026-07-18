#!/bin/bash
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass /data/lib/podman-data/projects/goml/libs/r1b_dk_wall > /tmp/dk_sass_pack.txt
echo "Function line:"
grep -n "Function : _ZN13fa_bwd_dk_new" /tmp/dk_sass_pack.txt
echo ""
echo "LDL/STL total count:"
grep -cE "\bLDL\b|\bSTL\b" /tmp/dk_sass_pack.txt
echo ""
echo "SHFL count:"
grep -cE "SHFL" /tmp/dk_sass_pack.txt
