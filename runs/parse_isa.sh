#!/bin/bash
# Parse isa_output.html: filter warnings, extract SASS mnemonics, group.
grep -v "Couldn't parse" /data/lib/podman-data/projects/goml/runs/isa_output.html \
    | grep -v "Cache loaded" \
    | grep -v "^$" \
    | sed 's/^@![PU]T* //' \
    | sed 's/^@!P[0-9]* //' \
    | sed 's/^@P[0-9]* //' \
    | sed 's/^@U[PT][0-9]* //' \
    | awk '{print $1}' \
    | sort | uniq -c | sort -rn
