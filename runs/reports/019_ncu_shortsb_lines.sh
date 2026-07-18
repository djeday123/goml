#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
DUMP=/tmp/dk_sass_source.txt

"$NCU" --import /tmp/dk_source.ncu-rep --page source > "$DUMP" 2>&1

echo "=== top-25 SASS instructions by stall_short_scoreboard (issued+not_issued) ==="
# Skip header, print addr + source + short_sb_issued + short_sb_notissued columns
# The table has fixed offsets; we grep any line that starts with 0x... and has SASS.
# Use awk on the stall_short_scoreboard column (need to find col index by header).
awk '
  /^0x/ && length($0) > 200 {
    # Extract short_sb (issued): column near end
    # Simpler: split by 2+ spaces to preserve numeric fields
    n = split($0, a, /  +/)
    # Find "0x..." and next fields
    addr = a[1]
    src  = a[2]
    # Numeric fields start after src; last ~30 are stall values.
    # stall_short_scoreboard (issued) is 15th from end, stall_short_scoreboard_not_issued is 15th column of not_issued block.
    # From the sample dump: 65 fields total, last two blocks of 16 stalls each.
    # Estimate: 33rd stall = short_sb; look at 34th from the end (not_issued) too.
    # simpler heuristic: sum of large numbers in tail
    tail_sum = 0
    for (i = n-30; i <= n; i++) tail_sum += a[i]+0
    # print if any stall > 100 aggregated
    if (tail_sum > 30) {
      printf "%s | %-60s | tail_sum=%d\n", addr, substr(src,1,60), tail_sum
    }
  }
' "$DUMP" | sort -t= -k2 -n -r | head -25
