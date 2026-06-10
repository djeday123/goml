#!/usr/bin/env python3
"""
v2: Proper live-range analysis with correct dst/src distinction.

Fix from v1: srcs do NOT include dst by default. Accumulator pattern (dst also
appearing in operands) properly handled — src read happens, THEN dst write
closes/reopens interval.

Output: per-register live intervals, max simultaneous live, and ranking by
"peak-overlap × (1/reads)" to find span-compression candidates.
"""

import re
from collections import defaultdict

SASS_LB3 = "/data/lib/podman-data/projects/goml/runs/sass_v89_lb3.txt"

inst_pat = re.compile(r'^\s*/\*([0-9a-fA-F]+)\*/\s+(\S.*?)(?:\s*;|\s*/\*)')
reg_pat = re.compile(r'\bR(\d+)\b')

STORE_MNEMS = {'STG', 'STS', 'STL', 'ATOMS', 'ATOM', 'RED', 'ST'}
NO_DEST_MNEMS = {'BAR', 'EXIT', 'NOP', 'BRA', 'BRX', 'CALL', 'JMP', 'CCTL',
                 'MEMBAR', 'SYNC', 'WARPSYNC', 'NANOSLEEP', 'WARPSYNC.ALL'}

def split_dest_src(body):
    """Return (dest_regs: set, src_regs: set). Properly handle accumulator."""
    parts = body.split(None, 1)
    if not parts:
        return set(), set()
    if body.startswith('@'):
        pp = body.split(None, 2)
        if len(pp) >= 2:
            body = ' '.join(pp[1:])
            parts = body.split(None, 1)
    mnem = parts[0]
    operand_str = parts[1] if len(parts) > 1 else ''
    all_regs = [int(g) for g in reg_pat.findall(operand_str)]
    base_mnem = mnem.split('.', 1)[0]
    if base_mnem in NO_DEST_MNEMS:
        return set(), set(all_regs)
    if base_mnem in STORE_MNEMS:
        return set(), set(all_regs)
    if not all_regs:
        return set(), set()
    dst = all_regs[0]
    srcs = set(all_regs[1:])  # rest of operands; dst may STILL appear (accumulator)
    return {dst}, srcs

# Parse SASS
instructions = []
inst_mnems = []
with open(SASS_LB3) as f:
    for raw in f:
        m = inst_pat.search(raw)
        if not m:
            continue
        addr = int(m.group(1), 16)
        body = m.group(2).strip()
        if body.startswith('@'):
            pp = body.split(None, 2)
            if len(pp) >= 2:
                body = ' '.join(pp[1:])
        dests, srcs = split_dest_src(body)
        dests = {r for r in dests if r < 254}
        srcs = {r for r in srcs if r < 254}
        instructions.append((addr, dests, srcs))
        inst_mnems.append(body.split(None, 1)[0])

n_inst = len(instructions)
mma_indices = [i for i, m in enumerate(inst_mnems) if 'HMMA' in m or 'QMMA' in m]
first_mma = mma_indices[0]
last_mma = mma_indices[-1]
print(f"Instructions: {n_inst}, addr range 0x{instructions[0][0]:04x}-0x{instructions[-1][0]:04x}")
print(f"First MMA: #{first_mma} @ 0x{instructions[first_mma][0]:04x}")
print(f"Last MMA:  #{last_mma} @ 0x{instructions[last_mma][0]:04x}")
print()

# Live-interval construction:
# For each reg, walk instructions. Each instruction:
#   - For each src: if reg is currently active (has prior write or is live-in),
#                   extend interval (update last_read = i, reads++).
#                   If not yet seen, mark live-in: start_idx = 0, reads = 1, last_read = i.
#   - For each dest: if reg currently active, close interval (start..last_read, reads).
#                    Start new interval: start_idx = i+1 (value will be used by SUBSEQUENT insts),
#                    reads = 0, last_read = i.

intervals_by_reg = defaultdict(list)
active_start = {}    # R# -> start idx of current interval
active_reads = {}    # R# -> read count in current interval
active_last_read = {}  # R# -> last read idx (or start if no reads)

for i, (addr, dests, srcs) in enumerate(instructions):
    # Process srcs first (they read the OLD value)
    for r in srcs:
        if r in active_start:
            active_last_read[r] = i
            active_reads[r] += 1
        else:
            # Live-in: born at function entry
            active_start[r] = 0
            active_last_read[r] = i
            active_reads[r] = 1
    # Process dests (they write a NEW value, closing old interval)
    for r in dests:
        if r in active_start:
            intervals_by_reg[r].append((active_start[r], active_last_read[r], active_reads[r]))
        active_start[r] = i
        active_last_read[r] = i
        active_reads[r] = 0

# Close remaining intervals
for r, start in active_start.items():
    intervals_by_reg[r].append((start, active_last_read[r], active_reads[r]))

# Compute per-instruction live reg count
live_at = [0] * n_inst
for r, ivs in intervals_by_reg.items():
    for s, e, _ in ivs:
        for i in range(s, e + 1):
            live_at[i] += 1

max_live = max(live_at)
max_live_idx = live_at.index(max_live)
print(f"MAX simultaneously-live: {max_live} at inst #{max_live_idx} (addr 0x{instructions[max_live_idx][0]:04x})")

# Distribution
buckets = [0]*10
for v in live_at:
    b = min(v // 20, 9)
    buckets[b] += 1
print("\nLive-count distribution:")
for i, c in enumerate(buckets):
    lo, hi = i*20, (i+1)*20-1
    label = f"[{lo:>3}-{hi:>3}]" if i < 9 else "[180+]   "
    print(f"  {label}: {c:>5} insts")

# Peak regions (live ≥ max_live - 5)
threshold = max_live - 5
print(f"\nPeak regions (live ≥ {threshold}):")
peak_regions = []
in_peak = False
ps = 0
for i, v in enumerate(live_at):
    if v >= threshold and not in_peak:
        ps = i
        in_peak = True
    elif v < threshold and in_peak:
        peak_regions.append((ps, i - 1))
        in_peak = False
if in_peak:
    peak_regions.append((ps, n_inst - 1))
print(f"  {len(peak_regions)} regions")
for ps, pe in peak_regions[:10]:
    pmax = max(live_at[ps:pe+1])
    is_in_loop = first_mma <= ps <= last_mma
    loc = "in-kv-loop" if is_in_loop else ("pre-loop" if pe < first_mma else "post-loop" if ps > last_mma else "boundary")
    print(f"  inst#{ps}-{pe}  (0x{instructions[ps][0]:04x}-0x{instructions[pe][0]:04x})  peak={pmax}  [{loc}]")
if len(peak_regions) > 10:
    print(f"  ... +{len(peak_regions)-10} more")

# Find intervals contributing to GLOBAL peak inst, ranked by low-use
print(f"\n=== Intervals active at GLOBAL peak (inst#{max_live_idx} @ 0x{instructions[max_live_idx][0]:04x}) ===")
peak_intervals = []
for r, ivs in intervals_by_reg.items():
    for s, e, reads in ivs:
        if s <= max_live_idx <= e:
            length = e - s + 1
            peak_intervals.append((r, s, e, length, reads))

peak_intervals.sort(key=lambda x: (x[4], -x[3]))
print(f"Intervals at peak: {len(peak_intervals)} (matches max_live)")

low_use_at_peak = [iv for iv in peak_intervals if iv[4] <= 2]
print(f"\nLOW-USE (≤2 reads) at peak: {len(low_use_at_peak)}")
print(f"  {'R#':>4}  {'start':>5}  {'end':>5}  {'len':>5}  {'reads':>5}")
for r, s, e, l, reads in low_use_at_peak[:30]:
    print(f"  R{r:<3}  0x{instructions[s][0]:04x}  0x{instructions[e][0]:04x}  {l:>5}  {reads:>5}")
if len(low_use_at_peak) > 30:
    print(f"  ... +{len(low_use_at_peak)-30} more")

print(f"\nMEDIUM-USE (3-5 reads) at peak: {sum(1 for iv in peak_intervals if 3 <= iv[4] <= 5)}")
print(f"HIGH-USE (≥6 reads) at peak:    {sum(1 for iv in peak_intervals if iv[4] >= 6)}")

# Final compression candidate scoring across ALL peak regions
print(f"\n=== Span-compression scoring (across ALL peak regions) ===")
interval_score = []
for r, ivs in intervals_by_reg.items():
    for s, e, reads in ivs:
        peak_overlap = 0
        for ps, pe in peak_regions:
            o_s = max(s, ps); o_e = min(e, pe)
            if o_s <= o_e:
                peak_overlap += o_e - o_s + 1
        if peak_overlap > 0:
            length = e - s + 1
            score = peak_overlap / max(reads, 1)
            interval_score.append((r, s, e, length, reads, peak_overlap, score))

interval_score.sort(key=lambda x: -x[6])

print(f"TOP 30 by peak_overlap / reads (best compression candidates):")
print(f"  {'R#':>4}  {'start':>6}  {'end':>6}  {'len':>5}  {'rd':>3}  {'pk-ov':>5}  {'score':>7}")
for r, s, e, l, reads, po, sc in interval_score[:30]:
    print(f"  R{r:<3}  0x{instructions[s][0]:04x}  0x{instructions[e][0]:04x}  {l:>5}  {reads:>3}  {po:>5}  {sc:>7.1f}")

# Verdict
print(f"\n=== Verdict ===")
n_low = len(low_use_at_peak)
if n_low >= 10:
    print(f"  {n_low} low-use intervals at GLOBAL peak. Span-compression PROMISING.")
    print(f"  → Potential reg savings: {n_low} (upper bound if all can be compressed)")
elif n_low >= 3:
    print(f"  {n_low} low-use intervals at peak — MODERATE pool. Try compressing top-3 first.")
else:
    print(f"  Only {n_low} low-use intervals at peak. Limited compression headroom.")
print(f"  Peak region count: {len(peak_regions)} — focused vs spread peak pressure matters too.")
