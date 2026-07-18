#!/bin/bash
REL=/data/lib/podman-data/projects/goml/release_v0.2.0
CLONE=/data/lib/podman-data/projects/goml/release_verify_clone
rm -rf "$CLONE"
mkdir -p "$CLONE/src" "$CLONE/tests" "$CLONE/docs/cert"
for F in "$REL"/src/*.cu "$REL"/src/*.cuh; do
    cp "$F" "$CLONE/src/"
done
for F in "$REL"/tests/*.cu; do
    cp "$F" "$CLONE/tests/"
done
cp "$REL/Makefile" "$CLONE/"
cp "$REL/verify.sh" "$CLONE/"
cp "$REL/README.md" "$CLONE/"
cp "$REL/LICENSE" "$CLONE/"
cp "$REL/docs/cert/cert_summary.md" "$CLONE/docs/cert/"
chmod +x "$CLONE/verify.sh"
echo "clone done:"
ls "$CLONE"
