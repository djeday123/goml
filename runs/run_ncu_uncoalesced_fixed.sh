#!/bin/bash
# UncoalescedSharedAccess properly parsed — v96 vs v111 vs v113 on PEAK cfg=9.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml

cd "$GOML"

profile_src() {
    local label="$1" bin="$2" cfg="$3"
    local out="$GOML/runs/ncu_${label}_uncoalesced_v2.csv"
    "$NCU" \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --section SourceCounters \
        --csv \
        "$bin" --ncu "$cfg" > "$out" 2>&1
    echo "--- $label ---"
    python3 << PYEOF
import csv, io, re
with open("$out") as f: lines = f.readlines()
hi = None
for i, l in enumerate(lines):
    if l.startswith('"ID","Process ID"'):
        hi = i; break
if hi is None:
    print("  (no header)")
    raise SystemExit(0)
rdr = csv.DictReader(io.StringIO(''.join(lines[hi:])))
for r in rdr:
    rule = r.get('Rule Name','')
    if 'Uncoalesced' in rule:
        desc = r.get('Rule Description','')
        speedup = r.get('Estimated Speedup','')
        # Try multiple patterns
        for pattern in [
            r'total of ([\d,]+) excessive (\w+) \(.*total ([\d,]+)',
            r'total of ([\d,]+) excessive (\w+).*total ([\d,]+) \w+',
        ]:
            m = re.search(pattern, desc)
            if m:
                exc = int(m.group(1).replace(',',''))
                tot = int(m.group(3).replace(',',''))
                unit = m.group(2)
                pct = 100.0*exc/tot if tot else 0
                print(f"  {rule:<28}  excessive {exc:>14,} / total {tot:>14,} {unit:<11}  {pct:5.1f}%   hint={speedup}")
                break
        else:
            # Print snippet if no match
            print(f"  {rule:<28}  (parse failed)  hint={speedup}")
            print(f"    desc[:200]: {desc[:200]}")
PYEOF
}

profile_src "v96"  "$GOML/runs/fa_v96_ksbatched"        9
profile_src "v111" "$GOML/runs/fa_v111_producer_skip"    9
profile_src "v113" "$GOML/runs/fa_v113_producer_arrive"  9

echo ""
echo "================================================================"
echo "If v96 ≈ v111 → baseline issue (smP STS / PV reads contribute heavily)"
echo "If v96 << v111 → consumer-stride transpose regression in v111"
echo "================================================================"
