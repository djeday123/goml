#!/bin/bash
# Liveness span analysis of v89 LB=3.
# For each register: first write addr, last read addr, span, uses-inside-span.
# Goal: find LONG-SPAN-LOW-USE regs = candidates for span compression via reorder.
# Empirical precedent: v87 reorder freed 8 regs without touching computation count.

set -uo pipefail

CUOBJ=/usr/local/cuda-13.1/bin/cuobjdump
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v89_pinregs_fp8"
SASS_FULL="$GOML/runs/sass_v89_full.txt"
SASS_LB3="$GOML/runs/sass_v89_lb3.txt"

cd "$GOML"

if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not found — build v89 first" >&2; exit 1; fi

echo "=== Step 1: dump SASS of v89 ==="
"$CUOBJ" --dump-sass "$BIN" > "$SASS_FULL" 2>&1
echo "wrote $SASS_FULL ($(wc -l < "$SASS_FULL") lines)"

echo ""
echo "=== Step 2: extract LB=3 kernel only ==="
LB3_FUNC="_Z11fa89_kernelILi3EEvPKhS1_S1_P6__halfiiifffi"
python3 << PYEOF
import re
with open("$SASS_FULL") as f: content = f.read()
fn_pat = re.compile(r"^\s*Function : ${LB3_FUNC}", re.MULTILINE)
m = fn_pat.search(content)
if not m:
    print("ERROR: could not find LB=3 function in SASS")
    raise SystemExit(1)
start = m.start()
next_fn = re.search(r"^\s*Function : ", content[m.end():], re.MULTILINE)
end = m.end() + next_fn.start() if next_fn else len(content)
with open("$SASS_LB3", 'w') as f: f.write(content[start:end])
print(f"wrote $SASS_LB3 ({content[start:end].count(chr(10))} lines)")
PYEOF

echo ""
echo "=== Step 3: liveness span analysis ==="
python3 << 'PYEOF'
import re
from collections import defaultdict

SASS_LB3 = "/data/lib/podman-data/projects/goml/runs/sass_v89_lb3.txt"

# Instruction pattern: /*XXXX*/   MNEMONIC  operands ;   /* encoding */
inst_pat = re.compile(r'^\s*/\*([0-9a-fA-F]+)\*/\s+(\S.*?)(?:\s*;|\s*/\*)')
reg_pat = re.compile(r'\bR(\d+)\b')

# Heuristic: for most instructions, dest is FIRST register listed after mnemonic.
# Exceptions: STG/STS/STL (store: first reg is source addr or value), CALL, BAR, EXIT.
STORE_MNEMS = {'STG', 'STS', 'STL', 'ATOMS', 'ATOM', 'RED', 'ST'}
NO_DEST_MNEMS = {'BAR', 'EXIT', 'NOP', 'BRA', 'BRX', 'CALL', 'JMP', 'CCTL',
                 'MEMBAR', 'SYNC', 'WARPSYNC', 'NANOSLEEP', 'CS2R'}
# CS2R writes — but its dest is also first reg. Keep simple.
# QMMA/HMMA: dest is first listed.

def split_dest_src(body):
    """Return (dest_regs_set, src_regs_set) — best-effort."""
    parts = body.split(None, 1)
    if not parts:
        return set(), set()
    mnem = parts[0]
    # Strip predicate prefix
    if mnem.startswith('@'):
        pp = body.split(None, 2)
        if len(pp) >= 2:
            mnem = pp[1]
            body = ' '.join(pp[1:])
            parts = body.split(None, 1)

    all_regs = [int(g) for g in reg_pat.findall(parts[1] if len(parts) > 1 else '')]

    base_mnem = mnem.split('.', 1)[0]
    if base_mnem in NO_DEST_MNEMS:
        return set(), set(all_regs)
    if base_mnem in STORE_MNEMS:
        # Store instructions write memory, all regs are sources
        return set(), set(all_regs)
    # Default: first reg is dest, rest are sources
    if not all_regs:
        return set(), set()
    return {all_regs[0]}, set(all_regs[1:])

# Per-register tracking
first_write = {}       # R# -> earliest write addr
last_read = {}         # R# -> latest read addr
first_appearance = {}  # R# -> first time seen (write or read)
last_appearance = {}   # R# -> last time seen
write_count = defaultdict(int)
read_count = defaultdict(int)
inst_addrs = []

with open(SASS_LB3) as f:
    for raw in f:
        m = inst_pat.search(raw)
        if not m:
            continue
        addr = int(m.group(1), 16)
        body = m.group(2).strip()
        # Strip predicate
        if body.startswith('@'):
            parts = body.split(None, 2)
            if len(parts) >= 2:
                body = ' '.join(parts[1:])
        dests, srcs = split_dest_src(body)
        inst_addrs.append(addr)
        for r in dests:
            if r not in first_write:
                first_write[r] = addr
            write_count[r] += 1
            if r not in first_appearance:
                first_appearance[r] = addr
            last_appearance[r] = addr
        for r in srcs:
            last_read[r] = addr  # always update — we want the LAST read
            read_count[r] += 1
            if r not in first_appearance:
                first_appearance[r] = addr
            last_appearance[r] = addr

# RZ (R255) is the zero register — skip
ALL_REGS = sorted(r for r in first_appearance.keys() if r < 254)

print(f"Total unique regs (excl. RZ): {len(ALL_REGS)}")
print(f"Instruction count: {len(inst_addrs)}")
print(f"First inst addr: 0x{inst_addrs[0]:04x}, Last: 0x{inst_addrs[-1]:04x}")
print(f"Total span:      {inst_addrs[-1] - inst_addrs[0]} bytes ({(inst_addrs[-1]-inst_addrs[0])//16} instructions @ 16B)")

# Build liveness summary
liveness = []
for r in ALL_REGS:
    fw = first_write.get(r, first_appearance.get(r, 0))
    lr = last_read.get(r, last_appearance.get(r, fw))
    fa = first_appearance.get(r, 0)
    la = last_appearance.get(r, 0)
    # Use overall first-touch and last-touch as span
    span = la - fa
    uses = read_count[r] + write_count[r]
    # Score: long span × low uses = high candidate
    # Use span/uses ratio (bytes-of-span per use, higher = more wasteful slot)
    score = span / max(uses, 1)
    liveness.append({
        'r': r, 'first': fa, 'last': la, 'span': span,
        'writes': write_count[r], 'reads': read_count[r], 'uses': uses,
        'score': score
    })

# Sort by score descending = best span-compression candidates
liveness.sort(key=lambda x: -x['score'])

print()
print("=== TOP 25 long-span-low-use registers (span-compression candidates) ===")
print(f"{'R#':>4}  {'first':>6}  {'last':>6}  {'span':>6}  {'W':>2}  {'R':>3}  {'uses':>4}  {'span/use':>9}")
for L in liveness[:25]:
    print(f"R{L['r']:<3}  0x{L['first']:04x}  0x{L['last']:04x}  {L['span']:>6}  "
          f"{L['writes']:>2}  {L['reads']:>3}  {L['uses']:>4}  {L['score']:>9.1f}")

# Categorize by liveness pattern
print()
print("=== Liveness category histogram ===")
total_span = inst_addrs[-1] - inst_addrs[0]
SHORT = total_span * 0.10
MEDIUM = total_span * 0.50

categories = {
    'short_span_any_use': [],          # short-lived: span < 10% of total
    'medium_span_low_use': [],         # medium span (10-50%) low use (≤3)
    'medium_span_high_use': [],        # medium span high use
    'long_span_low_use': [],           # long span (>50%) low use (≤5)
    'long_span_high_use': [],          # long span high use (recurrent / hot)
}
for L in liveness:
    if L['span'] < SHORT:
        categories['short_span_any_use'].append(L['r'])
    elif L['span'] < MEDIUM:
        if L['uses'] <= 3:
            categories['medium_span_low_use'].append(L['r'])
        else:
            categories['medium_span_high_use'].append(L['r'])
    else:
        if L['uses'] <= 5:
            categories['long_span_low_use'].append(L['r'])
        else:
            categories['long_span_high_use'].append(L['r'])

print(f"  short_span (any use) [<{SHORT/16:.0f} insts]:                      {len(categories['short_span_any_use'])} regs")
print(f"  medium_span low_use  [{SHORT/16:.0f}-{MEDIUM/16:.0f} insts, ≤3 uses]:        {len(categories['medium_span_low_use'])} regs")
print(f"  medium_span high_use [{SHORT/16:.0f}-{MEDIUM/16:.0f} insts, >3 uses]:        {len(categories['medium_span_high_use'])} regs")
print(f"  LONG_SPAN LOW_USE   [>{MEDIUM/16:.0f} insts, ≤5 uses]:           {len(categories['long_span_low_use'])} regs  ← compression candidates")
print(f"  long_span high_use   [>{MEDIUM/16:.0f} insts, >5 uses (recurrent)]: {len(categories['long_span_high_use'])} regs")

print()
print("=== LONG_SPAN LOW_USE register details (compression candidates) ===")
if categories['long_span_low_use']:
    for r in categories['long_span_low_use']:
        L = next(x for x in liveness if x['r'] == r)
        print(f"R{r:<3}  first=0x{L['first']:04x}  last=0x{L['last']:04x}  "
              f"span={L['span']:>5}  writes={L['writes']}  reads={L['reads']}  "
              f"uses={L['uses']}")
else:
    print("  NONE — all long-span regs are high-use (recurrent accumulators).")
    print("  → No span-compression opportunity at this level.")

print()
print("=== Verdict ===")
n_compress_candidates = len(categories['long_span_low_use'])
if n_compress_candidates >= 5:
    print(f"  {n_compress_candidates} long-span-low-use regs exist.")
    print("  → Span compression PATH OPEN. Try reordering creation/use to shrink span.")
    print("  → Expected: each compressed reg may free 0.5-1 slot in register allocator's")
    print("    maximum-simultaneous-live count.")
elif n_compress_candidates >= 1:
    print(f"  Only {n_compress_candidates} long-span-low-use regs found — marginal pool.")
    print("  → Span compression possible but gain ≤ a few regs at best.")
else:
    print("  NO long-span-low-use regs found.")
    print("  → Register allocator is ALREADY packing tightly. 4-block plan via span")
    print("    compression is also CLOSED — no idle slots to reclaim.")
PYEOF
