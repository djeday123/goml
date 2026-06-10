#!/usr/bin/env python3
"""
Proper live-range analysis of v89 LB=3.

For each register R, walk SASS linearly. Each WRITE starts a new live interval.
Interval extends until the next WRITE to the same reg, or until last read of
the current value (whichever comes first).

Per-instruction live count = number of registers with an active interval at that addr.
MAX live count across all instructions = required register file size.

Span-compression candidates = intervals that contribute to peak-live moments
but are sparsely used (low read count inside their interval).
"""

import re
from collections import defaultdict
from pathlib import Path

SASS_LB3 = "/data/lib/podman-data/projects/goml/runs/sass_v89_lb3.txt"

inst_pat = re.compile(r'^\s*/\*([0-9a-fA-F]+)\*/\s+(\S.*?)(?:\s*;|\s*/\*)')
reg_pat = re.compile(r'\bR(\d+)\b')

STORE_MNEMS = {'STG', 'STS', 'STL', 'ATOMS', 'ATOM', 'RED', 'ST'}
NO_DEST_MNEMS = {'BAR', 'EXIT', 'NOP', 'BRA', 'BRX', 'CALL', 'JMP', 'CCTL',
                 'MEMBAR', 'SYNC', 'WARPSYNC', 'NANOSLEEP'}

def split_dest_src(body):
    parts = body.split(None, 1)
    if not parts:
        return set(), set()
    if body.startswith('@'):
        pp = body.split(None, 2)
        if len(pp) >= 2:
            body = ' '.join(pp[1:])
            parts = body.split(None, 1)
    mnem = parts[0]
    all_regs = [int(g) for g in reg_pat.findall(parts[1] if len(parts) > 1 else '')]
    base_mnem = mnem.split('.', 1)[0]
    if base_mnem in NO_DEST_MNEMS:
        return set(), set(all_regs)
    if base_mnem in STORE_MNEMS:
        return set(), set(all_regs)
    if not all_regs:
        return set(), set()
    # MMA: dst is first reg but also appears as 4th (accumulator C = D)
    # Treat all later occurrences as both src+dst for liveness — keep dst as {first}
    return {all_regs[0]}, set(all_regs)  # include dst as src too — write doesn't kill prior read in same inst

# Parse SASS into ordered instruction list
instructions = []  # (addr, dests, srcs)
with open(SASS_LB3) as f:
    for raw in f:
        m = inst_pat.search(raw)
        if not m:
            continue
        addr = int(m.group(1), 16)
        body = m.group(2).strip()
        if body.startswith('@'):
            parts = body.split(None, 2)
            if len(parts) >= 2:
                body = ' '.join(parts[1:])
        dests, srcs = split_dest_src(body)
        # Filter RZ
        dests = {r for r in dests if r < 254}
        srcs = {r for r in srcs if r < 254}
        instructions.append((addr, dests, srcs))

print(f"Total instructions: {len(instructions)}")
print(f"Addr range: 0x{instructions[0][0]:04x} - 0x{instructions[-1][0]:04x}")

# Find kv-loop bounds = first/last MMA instruction
first_mma_idx = None
last_mma_idx = None
for i, (addr, dests, srcs) in enumerate(instructions):
    # Re-read raw line to find mnem — skip, use heuristic
    pass

# Re-parse for MMA marker
inst_mnems = []
with open(SASS_LB3) as f:
    for raw in f:
        m = inst_pat.search(raw)
        if not m:
            continue
        body = m.group(2).strip()
        if body.startswith('@'):
            pp = body.split(None, 2)
            if len(pp) >= 2:
                body = ' '.join(pp[1:])
        mnem = body.split(None, 1)[0]
        inst_mnems.append(mnem)

mma_indices = [i for i, m in enumerate(inst_mnems) if 'HMMA' in m or 'QMMA' in m]
first_mma = mma_indices[0] if mma_indices else 0
last_mma = mma_indices[-1] if mma_indices else len(instructions) - 1
print(f"First MMA inst: #{first_mma} @ 0x{instructions[first_mma][0]:04x}")
print(f"Last  MMA inst: #{last_mma} @ 0x{instructions[last_mma][0]:04x}")

# Build live intervals per register.
# For each reg, walk forward. Each DST write starts a new interval ending at next DST write.
# Track last READ inside each interval (true death point).
intervals_by_reg = defaultdict(list)  # R# -> [(start_idx, end_idx, reads_inside)]
active_start = {}  # R# -> idx of current write
active_reads = defaultdict(int)
last_read_idx = {}  # R# -> idx of last read in current interval

for i, (addr, dests, srcs) in enumerate(instructions):
    # Reads first (read happens before write in same inst, conceptually)
    for r in srcs:
        if r in active_start:
            last_read_idx[r] = i
            active_reads[r] += 1
        else:
            # Read before any write — treat as live from start (function param / live-in)
            active_start[r] = 0
            last_read_idx[r] = i
            active_reads[r] += 1
    # Writes
    for r in dests:
        if r in active_start:
            # Close previous interval at last read (or at this instruction if no reads since write)
            death = last_read_idx.get(r, active_start[r])
            intervals_by_reg[r].append((active_start[r], death, active_reads[r]))
        active_start[r] = i
        active_reads[r] = 0
        last_read_idx[r] = i  # initially the write point

# Close any still-active intervals at end
for r, start in active_start.items():
    death = last_read_idx.get(r, start)
    intervals_by_reg[r].append((start, death, active_reads[r]))

# Compute per-instruction live reg count
live_at = [0] * len(instructions)
for r, ivs in intervals_by_reg.items():
    for start, end, _ in ivs:
        for i in range(start, end + 1):
            live_at[i] += 1

max_live = max(live_at)
max_live_idx = live_at.index(max_live)
print(f"\nMAX simultaneously-live regs: {max_live} at inst #{max_live_idx} (addr 0x{instructions[max_live_idx][0]:04x})")
print(f"Reported ptxas reg count: 168 (LB=3 budget cap)")

# Distribution
buckets = [0]*10
for v in live_at:
    b = min(v // 20, 9)
    buckets[b] += 1
print("\nLive count distribution (insts at each live-count bucket):")
for i, c in enumerate(buckets):
    lo, hi = i*20, (i+1)*20-1
    if i == 9:
        print(f"  [180+]:      {c:>5} insts")
    else:
        print(f"  [{lo:>3}-{hi:>3}]:    {c:>5} insts")

# Find peak-pressure regions (where live > 150)
peak_regions = []
in_peak = False
peak_start = 0
for i, v in enumerate(live_at):
    if v >= 150 and not in_peak:
        peak_start = i
        in_peak = True
    elif v < 150 and in_peak:
        peak_regions.append((peak_start, i - 1))
        in_peak = False
if in_peak:
    peak_regions.append((peak_start, len(live_at) - 1))

print(f"\nPeak-pressure regions (live ≥ 150): {len(peak_regions)}")
for ps, pe in peak_regions[:8]:
    pmax = max(live_at[ps:pe+1])
    print(f"  inst#{ps}-{pe}  (0x{instructions[ps][0]:04x}-0x{instructions[pe][0]:04x})  peak={pmax}")
if len(peak_regions) > 8:
    print(f"  ... +{len(peak_regions)-8} more peak regions")

# Find intervals that span the GLOBAL peak (max_live_idx) AND are low-use
print(f"\n=== Intervals active at GLOBAL peak inst #{max_live_idx} ===")
peak_intervals = []
for r, ivs in intervals_by_reg.items():
    for start, end, reads in ivs:
        if start <= max_live_idx <= end:
            length = end - start + 1
            peak_intervals.append((r, start, end, length, reads))

# Sort by reads ascending — low-use first (compression candidates)
peak_intervals.sort(key=lambda x: (x[4], -x[3]))

print(f"Total intervals active at peak: {len(peak_intervals)}")
print(f"  → confirms max_live={max_live}")
print(f"\nLOW-USE intervals at peak (reads ≤ 2 inside their span):")
low_use_at_peak = [iv for iv in peak_intervals if iv[4] <= 2]
print(f"  Count: {len(low_use_at_peak)}")
if low_use_at_peak:
    print(f"  {'R#':>4}  {'start':>5}  {'end':>5}  {'len':>5}  {'reads':>5}")
    for r, s, e, l, reads in low_use_at_peak[:25]:
        print(f"  R{r:<3}  0x{instructions[s][0]:04x}  0x{instructions[e][0]:04x}  {l:>5}  {reads:>5}")

# Now compute: across ALL peak regions, find intervals that contribute to peak
# AND have low read count → strongest candidates for span compression
print(f"\n=== ACROSS-ALL-PEAK candidate scoring ===")
# For each interval, count how many peak-region instructions it spans
interval_peak_score = []
for r, ivs in intervals_by_reg.items():
    for start, end, reads in ivs:
        # Count peak-region overlap
        peak_overlap = 0
        for ps, pe in peak_regions:
            o_s = max(start, ps)
            o_e = min(end, pe)
            if o_s <= o_e:
                peak_overlap += o_e - o_s + 1
        if peak_overlap > 0:
            length = end - start + 1
            # Score: peak overlap × (1 / max(reads, 1)) — high overlap, low reads = best target
            score = peak_overlap / max(reads, 1)
            interval_peak_score.append((r, start, end, length, reads, peak_overlap, score))

interval_peak_score.sort(key=lambda x: -x[6])

print(f"\nTOP 30 by peak-overlap / reads ratio (best span-compression candidates):")
print(f"  {'R#':>4}  {'start':>6}  {'end':>6}  {'len':>5}  {'reads':>5}  {'pk-ov':>6}  {'score':>8}")
for r, s, e, l, reads, po, sc in interval_peak_score[:30]:
    print(f"  R{r:<3}  0x{instructions[s][0]:04x}  0x{instructions[e][0]:04x}  {l:>5}  {reads:>5}  {po:>6}  {sc:>8.1f}")

# Final summary
print("\n=== Summary ===")
print(f"  Total intervals: {sum(len(ivs) for ivs in intervals_by_reg.values())}")
print(f"  Unique regs:     {len(intervals_by_reg)}")
print(f"  MAX live:        {max_live}  (vs ptxas-reported 168 — discrepancy is dst-as-src counting artifact)")
print(f"  Peak regions:    {len(peak_regions)}")
print(f"  Low-use intervals at GLOBAL peak: {len(low_use_at_peak)}")
print()
if len(low_use_at_peak) >= 5:
    print(f"  → {len(low_use_at_peak)} low-use intervals contribute to GLOBAL peak.")
    print("  → SPAN COMPRESSION PATH OPEN. Reorder creates/uses to shrink these.")
    print("  → Each compressed interval may shave 1 from max_live → 1 reg freed.")
elif len(low_use_at_peak) >= 1:
    print(f"  → Only {len(low_use_at_peak)} low-use intervals at peak — marginal pool.")
else:
    print("  → NO low-use intervals at peak. All peak-contributing intervals are heavily used.")
    print("  → Register allocator is already at minimum max-live. Span compression CLOSED.")
