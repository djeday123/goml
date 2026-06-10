#!/bin/bash
# Level 1 ptxas ISA-spec probe — from user's old work
set -u

PTXAS="${PTXAS:-$(command -v ptxas || echo /usr/local/cuda-13.1/bin/ptxas)}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "ptxas: $PTXAS"
"$PTXAS" --version | head -1
echo ""

PTXVER="8.7"

emit_ptx() {
    local target="$1"; shift
    local body="$1"; shift
    cat <<EOF
.version $PTXVER
.target $target
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<4>;
    .reg .b32 %r<8>;
    .shared .align 1024 .b8 tmem_desc[16];
$body
    ld.param.u64 %rd0, [p];
    mov.u32 %r0, 1;
    st.global.u32 [%rd0], %r0;
    ret;
}
EOF
}

declare -A TESTS
TESTS[tcgen05_alloc]='    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem_desc], 32;'
TESTS[tcgen05_mma]='    tcgen05.mma.cta_group::1.kind::f16 [%rd1], %rd2, %rd3, %r4, 1;'
TESTS[tcgen05_ld]='    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r1}, [%rd1];'
TESTS[tcgen05_commit]='    tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [tmem_desc];'
TESTS[wgmma_fence]='    wgmma.fence.sync.aligned;'
TESTS[wgmma_commit]='    wgmma.commit_group.sync.aligned;'
TESTS[baseline]='    mov.u32 %r1, 42;'

TARGETS=(sm_90a sm_100a sm_120a)

printf "%-18s" "instruction"
for t in "${TARGETS[@]}"; do printf "%-12s" "$t"; done
echo ""
printf "%-18s" "------------------"
for t in "${TARGETS[@]}"; do printf "%-12s" "----------"; done
echo ""

for name in baseline wgmma_fence wgmma_commit tcgen05_alloc tcgen05_mma tcgen05_ld tcgen05_commit; do
    printf "%-18s" "$name"
    for target in "${TARGETS[@]}"; do
        ptx="$WORK/${name}_${target}.ptx"
        err="$WORK/${name}_${target}.err"
        emit_ptx "$target" "${TESTS[$name]}" > "$ptx"
        if "$PTXAS" -arch="$target" -o "$WORK/${name}_${target}.cubin" "$ptx" 2>"$err"; then
            printf "%-12s" "OK"
        else
            if grep -qi "not supported on .target" "$err"; then
                printf "%-12s" "NOT-IN-ISA"
            else
                printf "%-12s" "ERR(other)"
            fi
        fi
    done
    echo ""
done

echo ""
echo "Подробности ошибок sm_120a:"
for name in tcgen05_alloc tcgen05_mma tcgen05_commit wgmma_fence; do
    f="$WORK/${name}_sm_120a.err"
    [ -s "$f" ] && { echo "--- $name @ sm_120a ---"; head -2 "$f"; }
done
