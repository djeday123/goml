#!/bin/bash
export HOME=/tmp
NCU=/usr/local/cuda-13.1/bin/ncu
BIN=/data/lib/podman-data/projects/goml/libs/r1a_wall
OUT=/data/lib/podman-data/projects/goml/runs/reports/002_r1a_dram_ncu.csv

METRICS="\
dram__bytes.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors.sum,\
lts__t_bytes.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__t_bytes.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
smsp__cycles_active.avg,\
gpc__cycles_elapsed.avg"

"$NCU" --csv --launch-count 1 --launch-skip 5 --metrics "$METRICS" "$BIN" > "$OUT" 2>&1
echo "saved: $OUT"
