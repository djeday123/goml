#!/bin/bash
# Step 1: Source-attributed UncoalescedSharedAccess on v96 — answer the
# "which lines give the 40%" question. Zero code changes, one NCu pass.
#
# Profiling strategy:
#   --metrics memory_l1_wavefronts_shared / memory_l1_wavefronts_shared_excessive
#   per source line → identify hot site(s) responsible for excessive wavefronts
#
# Then categorise output: concentrated in 1-2 lines → targeted swizzle still
# possible. Spread across all SMEM ops → 40% may be FP8 layout floor.

set -uo pipefail

NCU=/usr/local/cuda-13.1/bin/ncu
GOML=/data/lib/podman-data/projects/goml
BIN="$GOML/runs/fa_v96_ksbatched"

cd "$GOML"

if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not built" >&2; exit 1; fi

REP="$GOML/runs/ncu_v96_uncoal_source.ncu-rep"

echo "=== Step 1: collect SourceCounters + L1 wavefront metrics on v96 cfg=9 ==="
"$NCU" \
    --target-processes all \
    --launch-skip 1 --launch-count 1 \
    --section SourceCounters \
    --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --import-source on \
    --export "$REP" \
    "$BIN" --ncu 9 2>&1 | tail -10

echo ""
echo "=== Source page dump → CSV ==="
SRC_CSV="$GOML/runs/ncu_v96_uncoal_source.csv"
"$NCU" --import "$REP" --page raw --csv --print-units base > "$SRC_CSV" 2>&1 || true
echo "Wrote $SRC_CSV ($(wc -l < $SRC_CSV) lines)"

echo ""
echo "=== Source page (text) ==="
SRC_TXT="$GOML/runs/ncu_v96_uncoal_source.txt"
"$NCU" --import "$REP" --page source > "$SRC_TXT" 2>&1 || true
echo "Wrote $SRC_TXT ($(wc -l < $SRC_TXT) lines)"

echo ""
echo "=== Details (bank-conflict ratio + speedup hint) ==="
"$NCU" --import "$REP" --page details 2>&1 | grep -A 3 -iE "uncoalesced|bank.conflict|wavefront" | head -40
