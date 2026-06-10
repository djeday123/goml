#!/bin/bash
# Compare UncoalescedSharedAccess on v96 (baseline) vs v111 (96-thread transpose) vs v113.
# If v96 also has ~40% → SMEM pattern is baseline issue, fix benefits ALL kernels.
# If v96 has lower (~10-20%) → our consumer-stride transpose introduces extra conflicts.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

profile() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_uncoalesced.csv"
    echo "================================================================"
    echo "  $label  cfg=$cfg"
    echo "================================================================"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --section SourceCounters \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
    python3 << PYEOF
with open("$out") as f: lines = f.readlines()
for l in lines:
    if 'UncoalescedSharedAccess' in l or 'UncoalescedGlobalAccess' in l:
        # Extract description value
        parts = l.split(',')
        if len(parts) > 14:
            desc = parts[17] if len(parts) > 17 else ''
            speedup = parts[19].strip().strip('"') if len(parts) > 19 else ''
            print(f"  {parts[11].strip().strip(chr(34))[:30]:<30} speedup hint: {speedup}")
            if 'wavefronts' in desc:
                # Try to extract numbers
                import re
                m = re.search(r'total of (\d[\d,]*) excessive wavefronts.*total (\d[\d,]*) wavefronts', desc)
                if m:
                    exc, tot = m.groups()
                    print(f"    excessive: {exc}  of total {tot}  ({100*int(exc.replace(',',''))/int(tot.replace(',',''))/1:.1f}%)")
                m2 = re.search(r'excessive sectors \((\d+)% of', desc)
                if m2:
                    print(f"    excessive global %: {m2.group(1)}%")
PYEOF
    echo ""
}

profile "v96"  "$GOML/runs/fa_v96_ksbatched"          9
profile "v111" "$GOML/runs/fa_v111_producer_skip"      9
profile "v113" "$GOML/runs/fa_v113_producer_arrive"    9

echo "================================================================"
echo "If v96 ≈ 10-15% and v111/v113 ≈ 35-40% → consumer-stride transpose introduces conflicts"
echo "If v96 also ≈ 30-40% → baseline issue (transpose pattern itself)"
echo "================================================================"
