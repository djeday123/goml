#!/bin/bash
# R03b FA-canary — 5-run fwd fingerprint against champion baseline v121r.
# Не пересертификация, канарейка после мостовых правок impl-1..3.
set -uo pipefail
BIN=/data/lib/podman-data/projects/goml/runs/fa_v121r_diet
if [ ! -x "$BIN" ]; then
    echo "MISSING binary: $BIN" >&2
    exit 2
fi

echo "=== warmup 10 launches ==="
for i in 1 2 3 4 5 6 7 8 9 10; do
    "$BIN" --time 128 8192 0 5 > /dev/null 2>&1
done
echo "warmup done"
echo ""

echo "=== FA-canary fwd v121r bh=128 sl=8192 wnd=0 — 5 measurement runs ==="
LOG=$(mktemp /tmp/canary_XXXXXX.log)
for run in 1 2 3 4 5; do
    OUT=$("$BIN" --time 128 8192 0 5 2>&1)
    TFLOPS=$(echo "$OUT" | grep -oE 'TFLOPS=[0-9.]+' | head -1 | tr -d 'TFLOPS=')
    MEDMS=$(echo "$OUT" | grep -oE 'median=[0-9.]+ms' | head -1 | tr -d 'median=ms')
    SMS=$(nvidia-smi --query-gpu=clocks.current.sm,temperature.gpu --format=csv,noheader,nounits | head -1)
    CLOCK=$(echo "$SMS" | awk -F, '{print $1}' | tr -d ' ')
    TEMP=$(echo "$SMS" | awk -F, '{print $2}' | tr -d ' ')
    printf "  run %d: %7.2f T  median=%sms  (sm_clock %sMHz, %sC)\n" \
           "$run" "$TFLOPS" "$MEDMS" "$CLOCK" "$TEMP"
    echo "$TFLOPS" >> "$LOG"
done
echo ""

echo "=== summary ==="
python3 - "$LOG" <<'PYEOF'
import sys, statistics
with open(sys.argv[1]) as f:
    vals = [float(x.strip()) for x in f if x.strip()]
med = statistics.median(vals)
mean = statistics.mean(vals)
print(f"  n={len(vals)}  median={med:.2f}T  mean={mean:.2f}T  min={min(vals):.2f}T  max={max(vals):.2f}T")
print(f"  expected: [652, 656] T (2026-08-03 A-LLM-6: recenter 652+-2 -> [652,656], факт. медианы 653.7-654.0)")
if 652 <= med <= 656:
    print("  VERDICT: WITHIN corridor")
else:
    print(f"  VERDICT: OUT OF corridor — STOP (delta from center 654 = {med - 654:+.2f}T)")
PYEOF
rm -f "$LOG"
