#!/bin/bash
/usr/local/cuda-12.8/bin/ptxas -arch=sm_120a -o /tmp/kb_check.cubin /tmp/kb.ptx 2>&1
echo "exit=$?"
