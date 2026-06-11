#!/bin/bash
CUDA=/usr/local/cuda-13.1
V3E=/data/lib/podman-data/projects/goml/runs/v3e.sass
V3H=/data/lib/podman-data/projects/goml/runs/v3h.sass
# Strip the leading filename header from cuobjdump and timestamps/path-deps
echo "=== Line-counts ==="
wc -l "$V3E" "$V3H"
echo ""
echo "=== Side-by-side opcode-only diff (offsets stripped) ==="
sed -E 's|/\*[0-9a-f]+\*/||; s|/\*[0-9a-f]+\*/||g; s|^\s+||' "$V3E" > /dev/shm/v3e.opcodes
sed -E 's|/\*[0-9a-f]+\*/||; s|/\*[0-9a-f]+\*/||g; s|^\s+||' "$V3H" > /dev/shm/v3h.opcodes
diff /dev/shm/v3e.opcodes /dev/shm/v3h.opcodes | head -40
echo ""
echo "=== Diff line-count: $(diff /dev/shm/v3e.opcodes /dev/shm/v3h.opcodes | wc -l) ==="
