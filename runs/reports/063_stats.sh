#!/bin/bash
FILE=/data/lib/podman-data/projects/goml/runs/reports/063_causal_30run.txt
grep "^run=" "$FILE" | awk -F'total=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/c_totals.txt
echo "N=$(wc -l < /tmp/c_totals.txt)"
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_total=%.4f\n", (v15+v16)/2}' /tmp/c_totals.txt
echo "min=$(head -1 /tmp/c_totals.txt) max=$(tail -1 /tmp/c_totals.txt)"
awk '{s+=$1;q+=$1*$1;n++} END{m=s/n;sd=sqrt(q/n-m*m); printf "mean=%.4f sd=%.4f CV=%.3f%%\n",m,sd,sd/m*100}' /tmp/c_totals.txt
echo "--- dk ---"
grep "^run=" "$FILE" | awk -F'dk_new=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/c_dk.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_dk=%.4f\n", (v15+v16)/2}' /tmp/c_dk.txt
echo "--- merged ---"
grep "^run=" "$FILE" | awk -F'merged=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/c_m.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_merged=%.4f\n", (v15+v16)/2}' /tmp/c_m.txt
echo "--- dq ---"
grep "^run=" "$FILE" | awk -F'dq_new=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/c_dq.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_dq=%.4f\n", (v15+v16)/2}' /tmp/c_dq.txt
echo "--- tflops16 ---"
grep "^run=" "$FILE" | awk -F'tflops16=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/c_tf.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_tflops16=%.4f\n", (v15+v16)/2}' /tmp/c_tf.txt
echo "--- temp ---"
grep "^run=" "$FILE" | grep -oE "temp=[0-9]+C" | sort -u | tr '\n' ' '; echo
