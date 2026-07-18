#!/bin/bash
FILE=/data/lib/podman-data/projects/goml/runs/reports/062_e2e_30run_nc.txt
grep "^run=" "$FILE" | awk -F'total=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/nc_totals.txt
echo "N=$(wc -l < /tmp/nc_totals.txt)"
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_total=%.4f\n", (v15+v16)/2}' /tmp/nc_totals.txt
echo "min=$(head -1 /tmp/nc_totals.txt) max=$(tail -1 /tmp/nc_totals.txt)"
awk '{s+=$1;q+=$1*$1;n++} END{m=s/n;sd=sqrt(q/n-m*m); printf "mean=%.4f sd=%.4f CV=%.3f%%\n",m,sd,sd/m*100}' /tmp/nc_totals.txt
echo "--- dk_new ---"
grep "^run=" "$FILE" | awk -F'dk_new=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/nc_dk.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_dk=%.4f\n", (v15+v16)/2}' /tmp/nc_dk.txt
echo "--- merged ---"
grep "^run=" "$FILE" | awk -F'merged=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/nc_merged.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_merged=%.4f\n", (v15+v16)/2}' /tmp/nc_merged.txt
echo "--- dq_new ---"
grep "^run=" "$FILE" | awk -F'dq_new=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/nc_dq.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_dq=%.4f\n", (v15+v16)/2}' /tmp/nc_dq.txt
echo "--- D ---"
grep "^run=" "$FILE" | awk -F'D=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/nc_D.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_D=%.4f\n", (v15+v16)/2}' /tmp/nc_D.txt
echo "--- tflops16 ---"
grep "^run=" "$FILE" | awk -F'tflops16=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/nc_tf16.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_tflops16=%.4f\n", (v15+v16)/2}' /tmp/nc_tf16.txt
echo "--- tflops10 ---"
grep "^run=" "$FILE" | awk -F'tflops10=' '{split($2, a, " "); print a[1]}' | sort -n > /tmp/nc_tf10.txt
awk 'NR==15{v15=$1} NR==16{v16=$1} END{printf "median_tflops10=%.4f\n", (v15+v16)/2}' /tmp/nc_tf10.txt
echo "--- temp range ---"
grep "^run=" "$FILE" | grep -oE "temp=[0-9]+C" | sort -u | tr '\n' ' '; echo
