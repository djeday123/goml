#!/bin/bash
python3 << 'PYEOF'
import re
with open('/data/lib/podman-data/projects/goml/runs/v96b.sass') as f:
    lines = f.readlines()
qmma_idx = [i for i,l in enumerate(lines) if 'QMMA' in l]
print(f"QMMA total: {len(qmma_idx)}, first at line {qmma_idx[0]+1}, last at line {qmma_idx[-1]+1}")
region = lines[qmma_idx[0]:qmma_idx[-1]+1]
print(f"Hot region span: {len(region)} lines")

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
print()
print("=== HOT LOOP counts (между первой и последней QMMA) ===")
addr = 0
for name, pat in patterns:
    rx = re.compile(pat)
    c = sum(1 for l in region if rx.search(l))
    print(f"  {name:<8}: {c:>4}")
    if name in ("IADD3","SHF.L","SHF.R","IMAD","LOP3","ISETP"):
        addr += c
print(f"  ADDR-сумма: {addr}")
PYEOF
