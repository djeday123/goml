#!/bin/bash
CUDA=/usr/local/cuda-13.1
SASS=/data/lib/podman-data/projects/goml/runs/v121.sass
"$CUDA/bin/cuobjdump" -sass /data/lib/podman-data/projects/goml/runs/fa_v121_addrhoist > "$SASS" 2>&1
wc -l "$SASS"

python3 << 'PYEOF'
import re
with open('/data/lib/podman-data/projects/goml/runs/v121.sass') as f:
    lines = f.readlines()
qmma_idx = [i for i,l in enumerate(lines) if 'QMMA' in l]
print(f"QMMA total: {len(qmma_idx)}, first {qmma_idx[0]+1}, last {qmma_idx[-1]+1}")
region = lines[qmma_idx[0]:qmma_idx[-1]+1]
print(f"Hot region: {len(region)} lines")
patterns = [
    ("IADD3",  r"\bIADD3\b"),
    ("SHF.L",  r"SHF\.L"),
    ("SHF.R",  r"SHF\.R"),
    ("IMAD",   r"\bIMAD\b"),
    ("LOP3",   r"\bLOP3\b"),
    ("ISETP",  r"\bISETP\b"),
    ("LDS",    r"\bLDS\b"),
    ("STS",    r"\bSTS\b"),
    ("PRMT",   r"\bPRMT\b"),
    ("QMMA",   r"\bQMMA\b"),
]
addr = 0
for name, pat in patterns:
    c = sum(1 for l in region if re.search(pat, l))
    print(f"  {name:<8}: {c:>4}")
    if name in ("IADD3","SHF.L","SHF.R","IMAD","LOP3","ISETP"):
        addr += c
print(f"  ADDR-сумма: {addr}")
print()
print("=== Сравнение с v96b ===")
print("  v96b ADDR-сумма : 268 (IADD3=9, SHF.R=7, IMAD=51, LOP3=24, ISETP=177)")
PYEOF
