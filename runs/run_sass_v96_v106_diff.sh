#!/bin/bash
# SASS diff v96 vs v106 hot loop. Research-only — measure instruction mix in
# PV hot region to spot what changed beyond reg/SMEM allocation.
#
# Counts: HMMA/QMMA (MMA), LDS (SMEM load), STS (SMEM store), LDSM (ldmatrix),
#         BAR (sync), CS2R (control), HFMA2/FMUL/FADD (alu), MOV, ISETP/BRA.

CUDA=/usr/local/cuda-13.1
GOML=/data/lib/podman-data/projects/goml
CUOBJDUMP=$CUDA/bin/cuobjdump

cd "$GOML"

dump_sass() {
    local label="$1" bin="$2"
    local out="$GOML/runs/sass_${label}.txt"
    "$CUOBJDUMP" --dump-sass "$bin" > "$out" 2>&1
    echo "  $label SASS lines: $(wc -l < "$out")"
}

count_mix() {
    local label="$1"
    local sass="$GOML/runs/sass_${label}.txt"
    python3 << PYEOF
import re
with open("$sass") as f: text = f.read()
# Filter to fa_kernel function body
m = re.search(r'Function : (_Z\d+fa\d+_kernel[^\n]+)\n(.+?)(?=\n\.headerflags|\Z)', text, re.DOTALL)
if not m:
    print("  ERROR: kernel symbol not found in $label"); raise SystemExit(0)
body = m.group(2)
patterns = {
    'HMMA/QMMA': r'\b(HMMA|QMMA|MMA)\b',
    'LDS':       r'\bLDS\b',
    'STS':       r'\bSTS\b',
    'LDSM':      r'\bLDSM\b',
    'BAR':       r'\bBAR\b',
    'HFMA2':     r'\bHFMA2\b',
    'FFMA':      r'\bFFMA\b',
    'FADD':      r'\bFADD\b',
    'FMUL':      r'\bFMUL\b',
    'MOV':       r'\bMOV\b',
    'IADD3':     r'\bIADD3\b',
    'ISETP':     r'\bISETP\b',
    'BRA':       r'\bBRA\b',
    'SHFL':      r'\bSHFL\b',
}
counts = {}
for k, p in patterns.items():
    counts[k] = len(re.findall(p, body))
total = sum(counts.values())
print(f"  --- {'$label':<8} | total {total} instructions in scope ---")
for k,v in sorted(counts.items(), key=lambda x:-x[1]):
    pct = (100.0*v/total) if total else 0
    print(f"    {k:<12} {v:>6}  {pct:>5.1f}%")
PYEOF
}

echo "=== Dumping SASS ==="
dump_sass "v96"  "$GOML/runs/fa_v96_ksbatched"
dump_sass "v106" "$GOML/runs/fa_v106_correct"

echo ""
echo "=== Instruction mix ==="
count_mix "v96"
count_mix "v106"

echo ""
echo "INTERPRETATION:"
echo "  More LDS+STS in v106 → SMEM port pressure (consistent with 12-warp contention)"
echo "  Same MMA count → confirms total compute is fixed; v106 just splits across more warps"
echo "  More SHFL in v106 → P-in-regs gather overhead per iter"
echo "  More BAR in v106 → transpose_v + extra sync structure"
