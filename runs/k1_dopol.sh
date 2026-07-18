#!/bin/bash
# K1-доп: wait class quantification per kernel
for kernel in dq dv dk; do
  csv=/data/lib/podman-data/projects/goml/runs/${kernel}_yarus2_src.csv
  if [ ! -f "$csv" ]; then echo "$kernel: MISSING $csv"; continue; fi
  echo ""
  echo "=== $kernel ==="
  sed -E 's/^"//; s/"$//' "$csv" | awk -F'","' '
    NR>2 {
      total += $48; totb += $32;
      if ($2 ~ /HMMA|QMMA/) mma += $48;
      else if ($2 ~ /BRA|BSYNC/) bra += $48;
      else if ($2 ~ /LDGSTS/) ldgsts += $48;
      else if ($2 ~ /LDS|LDCU|LDC /) lds += $48;
      else other += $48;
    }
    END {
      if (total==0) total=1;
      printf "  Total wait: %d samples\n", total;
      printf "  Total barrier: %d samples\n", totb;
      printf "  HMMA/QMMA wait: %d (%.1f%%)\n", mma, 100*mma/total;
      printf "  BRA/BSYNC wait: %d (%.1f%%)\n", bra, 100*bra/total;
      printf "  LDGSTS (cp.async drain) wait: %d (%.1f%%)\n", ldgsts, 100*ldgsts/total;
      printf "  LDS/LDC wait: %d (%.1f%%)\n", lds, 100*lds/total;
      printf "  Other (IADD/UIADD3/etc): %d (%.1f%%)\n", other, 100*other/total;
    }'
done
