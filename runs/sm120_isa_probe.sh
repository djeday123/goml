#!/bin/bash
# sm120_isa_probe.sh — exhaustive PTX feature coverage for sm_120a
# Probes ~50 major PTX instructions across all categories.

PTXAS="/usr/local/cuda-13.1/bin/ptxas"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PTXVER="8.7"
TARGET="sm_120a"

emit_ptx() {
    local body="$1"
    cat <<EOF
.version $PTXVER
.target $TARGET
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .pred %p<4>;
    .reg .b8 %b<4>;
    .reg .b16 %h<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .f64 %fd<4>;
    .shared .align 16 .b8 smem[1024];
    .shared .align 16 .b64 bar;
$body
    ret;
}
EOF
}

test_instr() {
    local name="$1"
    local body="$2"
    local ptx="$WORK/$name.ptx"
    local err="$WORK/$name.err"
    emit_ptx "$body" > "$ptx"
    if "$PTXAS" -arch="$TARGET" -o "$WORK/$name.cubin" "$ptx" 2>"$err"; then
        echo "OK"
    else
        if grep -qi "not supported on .target" "$err"; then
            echo "NOT-IN-ISA"
        elif grep -qi "Unknown modifier\|Illegal modifier\|Unknown instruction\|Could not find" "$err"; then
            echo "UNKNOWN"
        else
            echo "ERR(other)"
        fi
    fi
}

printf "%-40s %-15s\n" "instruction family / specific" "sm_120a"
printf "%-40s %-15s\n" "----------------------------------------" "----------"

# ===== MMA family (dense) =====
echo "--- MMA dense ---"
printf "%-40s %-15s\n" "mma m16n8k8 .f16" "$(test_instr "mma_m16k8_f16" '    mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%r1,%r2}, {%r3,%r4}, {%r5}, {%r6,%r7};')"
printf "%-40s %-15s\n" "mma m16n8k16 .f16" "$(test_instr "mma_m16k16_f16" '    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r1,%r2}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%r9,%r10};')"
printf "%-40s %-15s\n" "mma m16n8k16 .f32" "$(test_instr "mma_m16k16_f32" '    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0};')"
printf "%-40s %-15s\n" "mma m16n8k16 .bf16" "$(test_instr "mma_bf16" '    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0};')"
printf "%-40s %-15s\n" "mma m16n8k8 .tf32" "$(test_instr "mma_tf32" '    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0};')"
printf "%-40s %-15s\n" "mma m16n8k32 .e4m3 (FP8)" "$(test_instr "mma_fp8" '    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0};')"
printf "%-40s %-15s\n" "mma m16n8k32 .kind::f8f6f4 .e4m3" "$(test_instr "mma_kind_f8" '    mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0};')"

# ===== MMA Block scaled (Blackwell new) =====
echo "--- MMA block-scaled ---"
printf "%-40s %-15s\n" "mma m16n8k64 .kind::mxf4 (FP4)" "$(test_instr "mma_mxf4" '    mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0}, %r9, {0,0}, %r10, {0,0};')"
printf "%-40s %-15s\n" "mma m16n8k64 .kind::mxf4nvf4" "$(test_instr "mma_mxf4nvf4" '    mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0}, %r9, {0,0}, %r10, {0,0};')"
printf "%-40s %-15s\n" "mma m16n8k32 .kind::mxf8f6f4" "$(test_instr "mma_mxf8" '    mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8}, {%f5,%f6,%f7,%f0}, %r9, {0,0}, %r10, {0,0};')"

# ===== MMA Sparse =====
echo "--- MMA sparse ---"
printf "%-40s %-15s\n" "mma.sp m16n8k64 (FP8 sparse)" "$(test_instr "mma_sp_fp8" '    mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8,%r1,%r2}, {%f5,%f6,%f7,%f0}, %r9, 0x0;')"
printf "%-40s %-15s\n" "mma.sp::ordered_metadata FP4" "$(test_instr "mma_sp_ord_fp4" '    mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%f1,%f2,%f3,%f4}, {%r3,%r4,%r5,%r6}, {%r7,%r8,%r1,%r2}, {%f5,%f6,%f7,%f0}, %r9, 0x0, %r10, {0,0}, %r11, {0,0};')"

# ===== WGMMA (Hopper-only, should fail) =====
echo "--- WGMMA (Hopper async TC) ---"
printf "%-40s %-15s\n" "wgmma.fence" "$(test_instr "wgmma_fence" '    wgmma.fence.sync.aligned;')"
printf "%-40s %-15s\n" "wgmma.mma_async m64" "$(test_instr "wgmma_async" '    wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%f1,%f2,%f3,%f4}, %rd1, %rd2, 1, 1, 1, 1, 1;')"

# ===== TCGEN05 (DC Blackwell, should fail) =====
echo "--- TCGEN05 (5-gen DC Blackwell) ---"
printf "%-40s %-15s\n" "tcgen05.alloc" "$(test_instr "tcgen05_alloc" '    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem], 32;')"
printf "%-40s %-15s\n" "tcgen05.mma" "$(test_instr "tcgen05_mma" '    tcgen05.mma.cta_group::1.kind::f16 [%rd1], %rd2, %rd3, %r4, 1;')"

# ===== Memory ops =====
echo "--- Memory ops ---"
printf "%-40s %-15s\n" "cp.async.cg" "$(test_instr "cpa_cg" '    cp.async.cg.shared.global [smem], [%rd1], 16, 8;')"
printf "%-40s %-15s\n" "cp.async.ca" "$(test_instr "cpa_ca" '    cp.async.ca.shared.global [smem], [%rd1], 4, 4;')"
printf "%-40s %-15s\n" "cp.async.commit_group" "$(test_instr "cpa_commit" '    cp.async.commit_group;')"
printf "%-40s %-15s\n" "cp.async.wait_group" "$(test_instr "cpa_wait" '    cp.async.wait_group 0;')"
printf "%-40s %-15s\n" "cp.async.bulk (TMA)" "$(test_instr "tma_bulk" '    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [smem], [%rd1], 1024, [bar];')"
printf "%-40s %-15s\n" "cp.async.bulk.tensor (TMA tensor)" "$(test_instr "tma_tensor" '    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [smem], [%rd1, {%r1, %r2}], [bar];')"
printf "%-40s %-15s\n" "cp.async.bulk.commit_group" "$(test_instr "tma_commit" '    cp.async.bulk.commit_group;')"

# ===== mbarrier =====
echo "--- mbarrier ---"
printf "%-40s %-15s\n" "mbarrier.init" "$(test_instr "mbar_init" '    mbarrier.init.shared.b64 [bar], 1;')"
printf "%-40s %-15s\n" "mbarrier.arrive" "$(test_instr "mbar_arr" '    mbarrier.arrive.shared.b64 %rd1, [bar];')"
printf "%-40s %-15s\n" "mbarrier.test_wait" "$(test_instr "mbar_test" '    mbarrier.test_wait.shared.b64 %p1, [bar], %rd1;')"

# ===== ldmatrix / stmatrix =====
echo "--- ldmatrix / stmatrix ---"
printf "%-40s %-15s\n" "ldmatrix.x4 .b16" "$(test_instr "ldm4" '    ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r1,%r2,%r3,%r4}, [smem];')"
printf "%-40s %-15s\n" "ldmatrix.x4.trans .b16" "$(test_instr "ldm4t" '    ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%r1,%r2,%r3,%r4}, [smem];')"
printf "%-40s %-15s\n" "stmatrix.x4 .b16" "$(test_instr "stm4" '    stmatrix.sync.aligned.x4.m8n8.shared.b16 [smem], {%r1,%r2,%r3,%r4};')"
printf "%-40s %-15s\n" "stmatrix.x4.trans .b16" "$(test_instr "stm4t" '    stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [smem], {%r1,%r2,%r3,%r4};')"

# ===== Cluster / DSMEM (Hopper+) =====
echo "--- Cluster / DSMEM ---"
printf "%-40s %-15s\n" "%cluster_ctaid" "$(test_instr "cluster_ctaid" '    mov.u32 %r1, %cluster_ctaid.x;')"
printf "%-40s %-15s\n" "ld.shared::cluster" "$(test_instr "ld_cluster" '    ld.shared::cluster.u32 %r1, [smem];')"
printf "%-40s %-15s\n" "fence.sc.cluster" "$(test_instr "fence_cluster" '    fence.sc.cluster;')"

# ===== ex2 family (FP16 hardware exp) =====
echo "--- FP16 exp/MUFU ---"
printf "%-40s %-15s\n" "ex2.approx.f16x2" "$(test_instr "ex2_f16" '    ex2.approx.f16x2 %r1, %r2;')"
printf "%-40s %-15s\n" "ex2.approx.f32" "$(test_instr "ex2_f32" '    ex2.approx.f32 %f1, %f2;')"

# ===== FP8 conversion =====
echo "--- FP8 cvt ---"
printf "%-40s %-15s\n" "cvt.rn.satfinite.e4m3x2.f16x2" "$(test_instr "cvt_fp8" '    cvt.rn.satfinite.e4m3x2.f16x2 %h1, %r1;')"
printf "%-40s %-15s\n" "cvt.rn.satfinite.e5m2x2.f16x2" "$(test_instr "cvt_fp8_e5m2" '    cvt.rn.satfinite.e5m2x2.f16x2 %h1, %r1;')"

# ===== Atomics =====
echo "--- Atomics ---"
printf "%-40s %-15s\n" "atom.add.f16" "$(test_instr "atom_f16" '    atom.add.shared.f16 %h1, [smem], %h2;')"
printf "%-40s %-15s\n" "atom.add.f32" "$(test_instr "atom_f32" '    atom.add.shared.f32 %f1, [smem], %f2;')"
printf "%-40s %-15s\n" "atom.add.bf16" "$(test_instr "atom_bf16" '    atom.add.shared.bf16 %h1, [smem], %h2;')"

echo ""
echo "Legend:"
echo "  OK         = ptxas accepts → in ISA spec"
echo "  NOT-IN-ISA = 'not supported on .target sm_120a'"
echo "  UNKNOWN    = 'Unknown modifier/Unknown instruction'"
echo "  ERR(other) = syntax/operand error but opcode recognized"
