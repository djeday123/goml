#!/bin/bash
# 041 II.8 ABBA 8 пар: production dq (BASE) vs frozen d7a11a3d (CAND)
BASE=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e            # production dq_new (683396f8)
CAND=/data/lib/podman-data/projects/goml/libs/bench_r2c_e2e_dqfrozen   # frozen d7a11a3d (69r)
LOG=/data/lib/podman-data/projects/goml/runs/reports/041_dq_abba_data.txt
> "$LOG"

echo "=== fingerprint gate verification ===" | tee -a "$LOG"
"$BASE" 2>&1 | head -5 | tee -a "$LOG"
"$CAND" 2>&1 | head -5 | tee -a "$LOG"

echo "=== WARMUP (4 baseline runs, discarded) ==="
for i in 1 2 3 4; do "$BASE" 2>&1 | grep total > /dev/null; done

# ABBA schedule 8 pairs: B C C B B C C B B C C B B C C B
SEQ="B C C B B C C B B C C B B C C B"
i=0
echo "=== 041 II.8 ABBA 8 pairs (mode: in-chain, isolated timing) ===" | tee -a "$LOG"
for tag in $SEQ; do
    i=$((i+1))
    if [ "$tag" = "B" ]; then BIN=$BASE; NAME=BASE_prod; else BIN=$CAND; NAME=CAND_frozen; fi
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    LINE=$("$BIN" 2>&1 | grep "total=")
    echo "run=$i tag=$NAME temp=${TEMP}C $LINE" | tee -a "$LOG"
done
