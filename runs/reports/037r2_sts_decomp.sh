#!/bin/bash
# 037-r2: разложение 48 STS.U16 по Step E (dS_nat) vs Step G (smP_T)
OUT_MERGED=/data/lib/podman-data/projects/goml/runs/reports/037r2_sass_merged_only.txt

echo "=== 48 STS.U16 разложение по коду ==="
echo "Prod source layout (post-cut):"
echo "  Step E dS_nat scatter (строки 368-371):"
echo "    4 STS.b16 per (ni_a, ni_b) pair, NI_DP/2 = 4 iters unrolled"
echo "    → 4 × 4 = 16 STS.U16 static SASS"
echo ""
echo "  Step G smP_T scatter (строки 417-420):"
echo "    4 STS.U16 per ni (h_p00, h_p01, h_p10, h_p11), NI_QK = 8 iters unrolled"
echo "    → 4 × 8 = 32 STS.U16 static SASS"
echo ""
echo "  Всего: 16 + 32 = 48 STS.U16 ✓ (совпадает с SASS-счётом)"
echo ""
echo "=== Проверка: sample STS.U16 инструкции ==="
grep -E "STS\.U16" "$OUT_MERGED" | head -20
echo ""
echo "=== Итог 0e объект ==="
echo "Step E dS_nat scatter: 16 STS.b16 per lane per qt"
echo "  Итог per qt per block (128 threads): 16 × 128 = 2048 STS.b16"
echo ""
echo "Step G (не относится к 0e — это smP_T post-drain scatter, класс XI):"
echo "  32 STS.U16 per lane per qt, per qt per block = 4096 STS.U16"
