#!/bin/bash
# K4: "Other" wait class breakdown per kernel
# Split "other" into sub-classes: IADD/UIADD3, LEA, SHF, IMAD, MOV, other-alu
for kernel in dq dv dk; do
  csv=/data/lib/podman-data/projects/goml/runs/${kernel}_yarus2_src.csv
  if [ ! -f "$csv" ]; then continue; fi
  echo ""
  echo "=== $kernel: top-20 SASS by wait samples (any class) ==="
  sed -E 's/^"//; s/"$//' "$csv" | awk -F'","' 'NR>2 && $48+0 > 100 {printf "%d\t%s\n", $48, $2}' | sort -rn | head -20
  echo ""
  echo "  -- 'Other' breakdown --"
  sed -E 's/^"//; s/"$//' "$csv" | awk -F'","' '
    NR>2 && !($2 ~ /HMMA|QMMA|BRA|BSYNC|LDGSTS|LDS|LDCU|LDC / ) {
      if ($2 ~ /UIADD3/) uiadd += $48;
      else if ($2 ~ /IADD/) iadd += $48;
      else if ($2 ~ /LEA/) lea += $48;
      else if ($2 ~ /SHF/) shf += $48;
      else if ($2 ~ /IMAD/) imad += $48;
      else if ($2 ~ /MOV/) mov += $48;
      else if ($2 ~ /F2FP|CVT/) fcvt += $48;
      else if ($2 ~ /FMUL|FMA|FADD/) fp += $48;
      else if ($2 ~ /S2R|SR_/) s2r += $48;
      else oth += $48;
      total += $48;
    }
    END {
      if (total==0) total=1;
      printf "  UIADD3    : %d (%.1f%%) — predicated address arithmetic\n", uiadd, 100*uiadd/total;
      printf "  IADD      : %d (%.1f%%) — integer add (address chain)\n", iadd, 100*iadd/total;
      printf "  LEA       : %d (%.1f%%) — load effective address\n", lea, 100*lea/total;
      printf "  SHF       : %d (%.1f%%) — shift\n", shf, 100*shf/total;
      printf "  IMAD      : %d (%.1f%%) — integer multiply-add\n", imad, 100*imad/total;
      printf "  MOV       : %d (%.1f%%) — reg move\n", mov, 100*mov/total;
      printf "  F2FP/CVT  : %d (%.1f%%) — FP cast\n", fcvt, 100*fcvt/total;
      printf "  FP FMUL   : %d (%.1f%%) — FP arith\n", fp, 100*fp/total;
      printf "  S2R       : %d (%.1f%%) — special reg reads\n", s2r, 100*s2r/total;
      printf "  Other     : %d (%.1f%%)\n", oth, 100*oth/total;
    }'
done
