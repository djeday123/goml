#!/bin/bash
# Считаем integer-инструкции в теле QK/PV батчей v96b (baseline для v121).
# Подход:
#   1. cuobjdump -sass + поиск регионов между QMMA-блоками
#   2. Опираемся на -lineinfo: ищем lineinfo маркеры (.loc) chunks.
CUDA=/usr/local/cuda-13.1
BIN=/data/lib/podman-data/projects/goml/runs/fa_v96b_localfix
OUT=/data/lib/podman-data/projects/goml/runs/v96b.sass

"$CUDA/bin/cuobjdump" -sass "$BIN" > "$OUT" 2>&1
wc -l "$OUT"

echo ""
echo "=== Кол-во QMMA-инструкций (структура верхнего уровня) ==="
grep -cE 'QMMA' "$OUT"

echo ""
echo "=== Кол-во инструкций по классу во всём kernel ==="
echo "  IADD3   : $(grep -cE '\bIADD3\b' "$OUT")"
echo "  SHF.L   : $(grep -cE 'SHF\.L' "$OUT")"
echo "  SHF.R   : $(grep -cE 'SHF\.R' "$OUT")"
echo "  IMAD    : $(grep -cE '\bIMAD\b' "$OUT")"
echo "  LOP3    : $(grep -cE '\bLOP3\b' "$OUT")"
echo "  ISETP   : $(grep -cE '\bISETP\b' "$OUT")"
echo "  LDS     : $(grep -cE '\bLDS\b' "$OUT")"
echo "  STS     : $(grep -cE '\bSTS\b' "$OUT")"
echo "  XOR     : $(grep -cE '\bXOR\b' "$OUT")"
echo "  HMMA    : $(grep -cE '\bHMMA\b' "$OUT")"
echo "  QMMA    : $(grep -cE '\bQMMA\b' "$OUT")"
echo "  PRMT    : $(grep -cE '\bPRMT\b' "$OUT")"
