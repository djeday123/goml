#!/bin/bash
for MODE in nc causal; do
    FILE=/data/lib/podman-data/projects/goml/runs/reports/064_clone_30run_${MODE}.txt
    echo "=== $MODE (from clean clone) ==="
    grep "^run=" "$FILE" | awk -F'total=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/z.txt
    N=$(wc -l < /tmp/z.txt)
    echo "N=$N"
    awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_total=%.4f\n", (v15+v16)/2}' /tmp/z.txt
    awk '{s+=$1;q+=$1*$1;n++} END{m=s/n;sd=sqrt(q/n-m*m); printf "mean=%.4f sd=%.4f CV=%.3f%%\n",m,sd,sd/m*100}' /tmp/z.txt
    echo "min=$(head -1 /tmp/z.txt) max=$(tail -1 /tmp/z.txt)"
    grep "^run=" "$FILE" | awk -F'tflops16=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/tf.txt
    awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_tflops16=%.4f\n", (v15+v16)/2}' /tmp/tf.txt
    for K in D merged dk_new dq_new; do
        grep "^run=" "$FILE" | awk -F"${K}=" '{split($2, a, " "); print a[1]}' | sort -n > /tmp/k.txt
        awk -v K=$K 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_%s=%.4f\n", K, (v15+v16)/2}' /tmp/k.txt
    done
    echo ""
done
