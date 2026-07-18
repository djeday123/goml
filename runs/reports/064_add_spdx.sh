#!/bin/bash
# 064: prepend SPDX header to all publish sources in release_v0.2.0/
HEADER='// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Vugar and the FA-Blackwell-fp8 authors
'
REL=/data/lib/podman-data/projects/goml/release_v0.2.0

for F in $REL/src/*.cu $REL/src/*.cuh $REL/tests/*.cu; do
    if head -1 "$F" | grep -q "SPDX-License-Identifier"; then
        echo "skip (already has SPDX): $F"
        continue
    fi
    TMP="${F}.tmp"
    printf '%s' "$HEADER" > "$TMP"
    cat "$F" >> "$TMP"
    mv "$TMP" "$F"
    echo "added SPDX: $F"
done

echo "--- post-SPDX md5 ---"
md5sum $REL/src/*.cu $REL/src/*.cuh $REL/tests/*.cu
