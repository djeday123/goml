#!/bin/bash
CUDA=/usr/local/cuda-13.1
BIN=/data/lib/podman-data/projects/goml/runs/fa_v96b_localfix
echo "=== v96b FP32 instruction census ==="
N_FADD=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFADD\b')
N_FMUL=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFMUL\b')
N_FFMA=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bFFMA\b')
N_HFMA2=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bHFMA2\b')
N_HMUL2=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bHMUL2\b')
N_HADD2=$("$CUDA/bin/cuobjdump" -sass "$BIN" | grep -cE '\bHADD2\b')
echo "  FADD (FP32 add)  : $N_FADD"
echo "  FMUL (FP32 mul)  : $N_FMUL"
echo "  FFMA (FP32 fma)  : $N_FFMA"
echo "  ---"
echo "  HFMA2 (half2 fma): $N_HFMA2"
echo "  HMUL2 (half2 mul): $N_HMUL2"
echo "  HADD2 (half2 add): $N_HADD2"
echo ""
echo "=== First 30 FADD/FMUL locations ==="
"$CUDA/bin/cuobjdump" -sass "$BIN" | grep -nE '\bFADD\b|\bFMUL\b' | head -30
