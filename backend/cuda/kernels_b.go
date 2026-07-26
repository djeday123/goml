package cuda

// Phase B PTX kernels for complete training loop.
// Adds: abs, log, sqrt, tanh, relu, gelu, sigmoid (unary)
//       sub, div (binary)
//       sum_reduce, max_reduce (reductions)
//       layernorm (fused)
//       arange, where (utility)
//
// Same conventions as Phase A:
//   - sm_80+ target, FP32 only
//   - 256 threads/block
//   - %tidx/%bidx for thread/block indices
//   - $L_ prefix for labels
//   - exp(x) = 2^(x * log2(e)) via ex2.approx

const kernelPTX_B = `
.version 7.0
.target sm_80
.address_size 64

// ============================================================
// abs_f32: dst[i] = |src[i]|
// ============================================================
.visible .entry abs_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_abs_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %val, [%off];
    abs.f32 %val, %val;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_abs_done:
    ret;
}

// ============================================================
// log_f32: dst[i] = ln(src[i])
// ln(x) = log2(x) / log2(e) = log2(x) * (1/log2(e))
// 1/log2(e) = ln(2) = 0.693147... = 0x3F317218
// ============================================================
.visible .entry log_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %val, %ln2;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_log_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %val, [%off];
    lg2.approx.f32 %val, %val;
    mov.f32 %ln2, 0f3F317218;
    mul.f32 %val, %val, %ln2;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_log_done:
    ret;
}

// ============================================================
// sqrt_f32: dst[i] = sqrt(src[i])
// ============================================================
.visible .entry sqrt_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_sqrt_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %val, [%off];
    sqrt.approx.f32 %val, %val;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_sqrt_done:
    ret;
}

// ============================================================
// tanh_f32: dst[i] = tanh(src[i])
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
// ============================================================
.visible .entry tanh_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %x, %e2x, %one, %log2e, %two, %num, %den;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_tanh_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %x, [%off];

    mov.f32 %log2e, 0f3FB8AA3B;
    mov.f32 %one, 0f3F800000;
    mov.f32 %two, 0f40000000;
    mul.f32 %e2x, %x, %two;
    mul.f32 %e2x, %e2x, %log2e;
    ex2.approx.f32 %e2x, %e2x;
    sub.f32 %num, %e2x, %one;
    add.f32 %den, %e2x, %one;
    div.approx.f32 %x, %num, %den;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %x;
$L_tanh_done:
    ret;
}

// ============================================================
// relu_f32: dst[i] = max(0, src[i])
// ============================================================
.visible .entry relu_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %val, %zero;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_relu_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %val, [%off];
    mov.f32 %zero, 0f00000000;
    max.f32 %val, %val, %zero;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_relu_done:
    ret;
}

// ============================================================
// gelu_f32: dst[i] = 0.5*x*(1+tanh(c*(x+0.044715*x^3)))
// c = sqrt(2/pi) = 0.7978845608 = 0x3F4C422A
// 0.044715 = 0x3D372713
// ============================================================
.visible .entry gelu_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %x, %x3, %inner, %t, %e2t, %th, %half, %one;
    .reg .f32 %c, %k, %log2e, %two, %num, %den;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_gelu_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %x, [%off];

    mov.f32 %c, 0f3F4C422A;
    mov.f32 %k, 0f3D372713;
    mov.f32 %half, 0f3F000000;
    mov.f32 %one, 0f3F800000;
    mov.f32 %two, 0f40000000;
    mov.f32 %log2e, 0f3FB8AA3B;

    // x^3
    mul.f32 %x3, %x, %x;
    mul.f32 %x3, %x3, %x;
    // inner = c * (x + 0.044715 * x^3)
    mul.f32 %inner, %k, %x3;
    add.f32 %inner, %x, %inner;
    mul.f32 %inner, %c, %inner;
    // tanh(inner) = (exp(2*inner)-1)/(exp(2*inner)+1)
    mul.f32 %t, %inner, %two;
    mul.f32 %t, %t, %log2e;
    ex2.approx.f32 %e2t, %t;
    sub.f32 %num, %e2t, %one;
    add.f32 %den, %e2t, %one;
    div.approx.f32 %th, %num, %den;
    // result = 0.5 * x * (1 + tanh)
    add.f32 %th, %one, %th;
    mul.f32 %th, %x, %th;
    mul.f32 %th, %half, %th;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %th;
$L_gelu_done:
    ret;
}

// ============================================================
// sigmoid_f32: dst[i] = 1 / (1 + exp(-src[i]))
// ============================================================
.visible .entry sigmoid_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %x, %t, %one, %log2e;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_sig_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %x, [%off];

    mov.f32 %log2e, 0f3FB8AA3B;
    mov.f32 %one, 0f3F800000;
    neg.f32 %t, %x;
    mul.f32 %t, %t, %log2e;
    ex2.approx.f32 %t, %t;
    add.f32 %t, %t, %one;
    rcp.approx.f32 %t, %t;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %t;
$L_sig_done:
    ret;
}

// ============================================================
// sub_f32: dst[i] = a[i] - b[i]
// ============================================================
.visible .entry sub_f32(
    .param .u64 p_dst,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %a, %b, %off;
    .reg .f32 %va, %vb;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_sub_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %a, %off;
    ld.global.f32 %va, [%off];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %b, %off;
    ld.global.f32 %vb, [%off];
    sub.f32 %va, %va, %vb;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %va;
$L_sub_done:
    ret;
}

// ============================================================
// div_f32: dst[i] = a[i] / b[i]
// ============================================================
.visible .entry div_f32(
    .param .u64 p_dst,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %a, %b, %off;
    .reg .f32 %va, %vb;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_div_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %a, %off;
    ld.global.f32 %va, [%off];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %b, %off;
    ld.global.f32 %vb, [%off];
    div.approx.f32 %va, %va, %vb;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %va;
$L_div_done:
    ret;
}

// ============================================================
// sum_reduce_f32: per-row sum reduction
// Grid: (num_rows, 1, 1), Block: (256, 1, 1)
// Each block reduces one row of row_size elements.
// dst[row] = sum(src[row*row_size .. (row+1)*row_size])
// ============================================================
.visible .entry sum_reduce_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_row_size,
    .param .u32 p_num_rows
) {
    .reg .u32 %tidx, %bidx, %row_size, %num_rows, %i, %row_off;
    .reg .u64 %dst, %src, %base, %off;
    .reg .f32 %acc, %val;
    .reg .pred %p, %ploop;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %row_size, [p_row_size];
    ld.param.u32 %num_rows, [p_num_rows];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    setp.ge.u32 %p, %bidx, %num_rows;
    @%p bra $L_sum_done;

    // base = src + bidx * row_size * 4
    mul.lo.u32 %row_off, %bidx, %row_size;
    mul.wide.u32 %base, %row_off, 4;
    add.u64 %base, %src, %base;

    // Accumulate with stride 256
    mov.f32 %acc, 0f00000000;
    mov.u32 %i, %tidx;
$L_sum_loop:
    setp.ge.u32 %ploop, %i, %row_size;
    @%ploop bra $L_sum_loop_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %base, %off;
    ld.global.f32 %val, [%off];
    add.f32 %acc, %acc, %val;
    add.u32 %i, %i, 256;
    bra $L_sum_loop;
$L_sum_loop_end:

    // For correctness with 256 threads, only thread 0 writes.
    // Each thread has a partial sum; we need warp/block reduction.
    // Simple approach: thread 0 does full serial sum.
    // TODO: warp shuffle reduction for perf.
    setp.ne.u32 %p, %tidx, 0;
    @%p bra $L_sum_done;

    // Thread 0 re-scans entire row (simple but correct)
    mov.f32 %acc, 0f00000000;
    mov.u32 %i, 0;
$L_sum_serial:
    setp.ge.u32 %ploop, %i, %row_size;
    @%ploop bra $L_sum_serial_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %base, %off;
    ld.global.f32 %val, [%off];
    add.f32 %acc, %acc, %val;
    add.u32 %i, %i, 1;
    bra $L_sum_serial;
$L_sum_serial_end:

    // Store result
    mul.wide.u32 %off, %bidx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %acc;

$L_sum_done:
    ret;
}

// ============================================================
// max_reduce_f32: per-row max reduction
// Grid: (num_rows, 1, 1), Block: (256, 1, 1)
// dst[row] = max(src[row*row_size .. (row+1)*row_size])
// ============================================================
.visible .entry max_reduce_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_row_size,
    .param .u32 p_num_rows
) {
    .reg .u32 %tidx, %bidx, %row_size, %num_rows, %i, %row_off;
    .reg .u64 %dst, %src, %base, %off;
    .reg .f32 %acc, %val;
    .reg .pred %p, %ploop;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %row_size, [p_row_size];
    ld.param.u32 %num_rows, [p_num_rows];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    setp.ge.u32 %p, %bidx, %num_rows;
    @%p bra $L_max_done;

    mul.lo.u32 %row_off, %bidx, %row_size;
    mul.wide.u32 %base, %row_off, 4;
    add.u64 %base, %src, %base;

    // Thread 0 serial scan (correct, perf via warp shuffle later)
    setp.ne.u32 %p, %tidx, 0;
    @%p bra $L_max_done;

    mov.f32 %acc, 0fFF800000;
    mov.u32 %i, 0;
$L_max_loop:
    setp.ge.u32 %ploop, %i, %row_size;
    @%ploop bra $L_max_loop_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %base, %off;
    ld.global.f32 %val, [%off];
    max.f32 %acc, %acc, %val;
    add.u32 %i, %i, 1;
    bra $L_max_loop;
$L_max_loop_end:

    mul.wide.u32 %off, %bidx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %acc;

$L_max_done:
    ret;
}

// ============================================================
// layernorm_f32: fused LayerNorm
// y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
// Grid: (num_rows, 1, 1), Block: (256, 1, 1)
// Each block normalizes one row of norm_size elements.
// gamma, beta: [norm_size] (shared across rows)
// ============================================================
.visible .entry layernorm_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u64 p_gamma,
    .param .u64 p_beta,
    .param .u32 p_norm_size,
    .param .u32 p_num_rows,
    .param .f32 p_eps
) {
    .reg .u32 %tidx, %bidx, %norm_size, %num_rows, %i, %row_off;
    .reg .u64 %dst, %src, %gamma, %beta, %base, %off, %goff;
    .reg .f32 %val, %mean, %var, %eps, %inv_n, %diff, %inv_std;
    .reg .f32 %g, %b, %sum, %sum2;
    .reg .pred %p, %ploop;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u64 %gamma, [p_gamma];
    ld.param.u64 %beta, [p_beta];
    ld.param.u32 %norm_size, [p_norm_size];
    ld.param.u32 %num_rows, [p_num_rows];
    ld.param.f32 %eps, [p_eps];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    setp.ge.u32 %p, %bidx, %num_rows;
    @%p bra $L_ln_done;

    // Only thread 0 computes (correct, perf later via parallel reduction)
    setp.ne.u32 %p, %tidx, 0;
    @%p bra $L_ln_done;

    // base = src + bidx * norm_size * 4
    mul.lo.u32 %row_off, %bidx, %norm_size;
    mul.wide.u32 %base, %row_off, 4;
    add.u64 %base, %src, %base;

    // Pass 1: compute mean
    mov.f32 %sum, 0f00000000;
    mov.u32 %i, 0;
$L_ln_mean:
    setp.ge.u32 %ploop, %i, %norm_size;
    @%ploop bra $L_ln_mean_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %base, %off;
    ld.global.f32 %val, [%off];
    add.f32 %sum, %sum, %val;
    add.u32 %i, %i, 1;
    bra $L_ln_mean;
$L_ln_mean_end:

    cvt.rn.f32.u32 %inv_n, %norm_size;
    rcp.approx.f32 %inv_n, %inv_n;
    mul.f32 %mean, %sum, %inv_n;

    // Pass 2: compute variance
    mov.f32 %sum2, 0f00000000;
    mov.u32 %i, 0;
$L_ln_var:
    setp.ge.u32 %ploop, %i, %norm_size;
    @%ploop bra $L_ln_var_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %base, %off;
    ld.global.f32 %val, [%off];
    sub.f32 %diff, %val, %mean;
    mul.f32 %diff, %diff, %diff;
    add.f32 %sum2, %sum2, %diff;
    add.u32 %i, %i, 1;
    bra $L_ln_var;
$L_ln_var_end:

    mul.f32 %var, %sum2, %inv_n;
    add.f32 %var, %var, %eps;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;

    // Pass 3: normalize, scale, shift, store
    // dst_base = dst + bidx * norm_size * 4
    mul.wide.u32 %off, %row_off, 4;
    add.u64 %off, %dst, %off;
    // save dst base in %base (reuse)
    mov.u64 %base, %off;
    // reload src base
    mul.wide.u32 %off, %row_off, 4;
    add.u64 %off, %src, %off;

    mov.u32 %i, 0;
$L_ln_norm:
    setp.ge.u32 %ploop, %i, %norm_size;
    @%ploop bra $L_ln_norm_end;
    // Load src
    mul.wide.u32 %goff, %i, 4;
    add.u64 %goff, %off, %goff;
    ld.global.f32 %val, [%goff];
    // normalize
    sub.f32 %val, %val, %mean;
    mul.f32 %val, %val, %inv_std;
    // Load gamma, beta
    mul.wide.u32 %goff, %i, 4;
    add.u64 %goff, %gamma, %goff;
    ld.global.f32 %g, [%goff];
    mul.wide.u32 %goff, %i, 4;
    add.u64 %goff, %beta, %goff;
    ld.global.f32 %b, [%goff];
    // y = gamma * normalized + beta
    mul.f32 %val, %val, %g;
    add.f32 %val, %val, %b;
    // Store
    mul.wide.u32 %goff, %i, 4;
    add.u64 %goff, %base, %goff;
    st.global.f32 [%goff], %val;
    add.u32 %i, %i, 1;
    bra $L_ln_norm;
$L_ln_norm_end:

$L_ln_done:
    ret;
}

// ============================================================
// arange_f32: dst[i] = start + i * step
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// ============================================================
.visible .entry arange_f32(
    .param .u64 p_dst,
    .param .f32 p_start,
    .param .f32 p_step,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %off;
    .reg .f32 %start, %step, %fi, %val;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.f32 %start, [p_start];
    ld.param.f32 %step, [p_step];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_arange_done;

    cvt.rn.f32.u32 %fi, %idx;
    mul.f32 %val, %fi, %step;
    add.f32 %val, %val, %start;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_arange_done:
    ret;
}

// ============================================================
// where_f32: dst[i] = cond[i] != 0 ? a[i] : b[i]
// cond is float32, nonzero = true
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// ============================================================
.visible .entry where_f32(
    .param .u64 p_dst,
    .param .u64 p_cond,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %cond, %a, %b, %off;
    .reg .f32 %vc, %va, %vb, %zero, %res;
    .reg .pred %p, %pcond;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %cond, [p_cond];
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_where_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %cond, %off;
    ld.global.f32 %vc, [%off];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %a, %off;
    ld.global.f32 %va, [%off];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %b, %off;
    ld.global.f32 %vb, [%off];

    mov.f32 %zero, 0f00000000;
    setp.ne.f32 %pcond, %vc, %zero;
    selp.f32 %res, %va, %vb, %pcond;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %res;
$L_where_done:
    ret;
}

// ============================================================
// cross_entropy_f32: fused CrossEntropy forward + backward (A-1, 2026-07-24).
// Input:
//   logits    [num_rows, vocab]  F32
//   targets   [num_rows]         int32 (uploaded as u32-compat, indices >= 0)
// Output:
//   loss_out  [num_rows]         F32  per-row NLL (unscaled -- host sum/m)
//   grad_out  [num_rows, vocab]  F32  (softmax - onehot) * inv_bs
// Const scalar: inv_bs (F32) -- usually 1.0 / num_rows.
//
// Grid: (num_rows, 1, 1), Block: (256, 1, 1). Thread 0 per block does the work
// (same pattern as softmax_f32; V=32000 sequential in 3 passes, ~0.3-1 ms total
// for 1024 rows in parallel across SMs).
// Two-pass: (max) + (sum exp) + (loss + grad). log/exp via lg2.approx.f32/ex2.approx.f32.
// ============================================================
.visible .entry cross_entropy_f32(
    .param .u64 p_logits,
    .param .u64 p_targets,
    .param .u64 p_loss_out,
    .param .u64 p_grad_out,
    .param .u32 p_num_rows,
    .param .u32 p_vocab,
    .param .f32 p_inv_bs
) {
    .reg .u32 %tidx, %bidx, %num_rows, %vocab, %i, %row_off_u32, %tgt;
    .reg .u64 %logits, %targets, %loss_out, %grad_out;
    .reg .u64 %logits_base, %grad_base, %off;
    .reg .f32 %val, %max_val, %sum, %log2e, %ln2, %inv_sum;
    .reg .f32 %inv_bs, %e, %p_prob, %log_sum, %log_z, %tgt_logit, %loss;
    .reg .f32 %one;
    .reg .pred %pred, %ploop;

    ld.param.u64 %logits,   [p_logits];
    ld.param.u64 %targets,  [p_targets];
    ld.param.u64 %loss_out, [p_loss_out];
    ld.param.u64 %grad_out, [p_grad_out];
    ld.param.u32 %num_rows, [p_num_rows];
    ld.param.u32 %vocab,    [p_vocab];
    ld.param.f32 %inv_bs,   [p_inv_bs];

    mov.u32 %bidx, %ctaid.x;
    setp.ge.u32 %pred, %bidx, %num_rows;
    @%pred bra $L_ce_done;

    // Only thread 0 in the block does the work.
    mov.u32 %tidx, %tid.x;
    setp.ne.u32 %pred, %tidx, 0;
    @%pred bra $L_ce_done;

    // logits_base = logits + bidx * vocab * 4
    mul.lo.u32 %row_off_u32, %bidx, %vocab;
    mul.wide.u32 %logits_base, %row_off_u32, 4;
    add.u64 %logits_base, %logits, %logits_base;
    // grad_base = grad_out + bidx * vocab * 4
    mul.wide.u32 %grad_base, %row_off_u32, 4;
    add.u64 %grad_base, %grad_out, %grad_base;

    // Pass 1: find max
    mov.f32 %max_val, 0fFF800000;    // -inf
    mov.u32 %i, 0;
$L_ce_max:
    setp.ge.u32 %ploop, %i, %vocab;
    @%ploop bra $L_ce_max_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %logits_base, %off;
    ld.global.f32 %val, [%off];
    max.f32 %max_val, %max_val, %val;
    add.u32 %i, %i, 1;
    bra $L_ce_max;
$L_ce_max_end:

    // Pass 2: sum = sum_i exp(logits[i] - max_val)
    // exp(x) = 2^(x * log2e). log2e = 0x3FB8AA3B.
    mov.f32 %sum, 0f00000000;
    mov.f32 %log2e, 0f3FB8AA3B;
    mov.u32 %i, 0;
$L_ce_sum:
    setp.ge.u32 %ploop, %i, %vocab;
    @%ploop bra $L_ce_sum_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %logits_base, %off;
    ld.global.f32 %val, [%off];
    sub.f32 %val, %val, %max_val;
    mul.f32 %val, %val, %log2e;
    ex2.approx.f32 %val, %val;
    add.f32 %sum, %sum, %val;
    add.u32 %i, %i, 1;
    bra $L_ce_sum;
$L_ce_sum_end:

    // log_z = max_val + ln(sum) = max_val + lg2.approx.f32(sum) * ln(2).
    // ln(2) = 0x3F317218.
    mov.f32 %ln2, 0f3F317218;
    lg2.approx.f32 %log_sum, %sum;
    mul.f32 %log_sum, %log_sum, %ln2;
    add.f32 %log_z, %max_val, %log_sum;

    // Load target (interpret int32 as u32; targets >= 0 by contract).
    mul.wide.u32 %off, %bidx, 4;
    add.u64 %off, %targets, %off;
    ld.global.u32 %tgt, [%off];

    // Load logits[bidx, tgt].
    mul.wide.u32 %off, %tgt, 4;
    add.u64 %off, %logits_base, %off;
    ld.global.f32 %tgt_logit, [%off];

    // loss = log_z - tgt_logit  (per-row NLL, unscaled).
    sub.f32 %loss, %log_z, %tgt_logit;
    mul.wide.u32 %off, %bidx, 4;
    add.u64 %off, %loss_out, %off;
    st.global.f32 [%off], %loss;

    // Pass 3: grad[i] = (exp(logits[i]-max)/sum - onehot[i]) * inv_bs.
    rcp.approx.f32 %inv_sum, %sum;
    mov.f32 %one, 0f3F800000;        // 1.0
    mov.u32 %i, 0;
$L_ce_grad:
    setp.ge.u32 %ploop, %i, %vocab;
    @%ploop bra $L_ce_grad_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %logits_base, %off;
    ld.global.f32 %val, [%off];
    sub.f32 %val, %val, %max_val;
    mul.f32 %val, %val, %log2e;
    ex2.approx.f32 %e, %val;
    mul.f32 %p_prob, %e, %inv_sum;
    setp.eq.u32 %pred, %i, %tgt;
    @%pred sub.f32 %p_prob, %p_prob, %one;
    mul.f32 %p_prob, %p_prob, %inv_bs;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %grad_base, %off;
    st.global.f32 [%off], %p_prob;
    add.u32 %i, %i, 1;
    bra $L_ce_grad;
$L_ce_grad_end:

$L_ce_done:
    ret;
}

// ============================================================
// transpose_shd_hsd_f32 (A-LLM-1 Stage 3, 2026-07-25):
//   Permute [S, H, hd] -> [H, S, hd] on device.
//   Each (h, s) pair -- one block; thread copies one hd element.
//   Layout FA: attention needs [B*H, S, hd] bh-major; after Wq/Wk/Wv
//   output [S, H*hd] naturally viewed as [S, H, hd]; permute gives [H, S, hd].
//   For B>1: caller does per-batch permute into different bh-offsets scratch.
//
// Grid: (H, S, 1). Block: (hd, 1, 1). hd in {64, 128}.
// src[s, h, d] at src + (s*H*hd + h*hd + d) * 4
// dst[h, s, d] at dst + (h*S*hd + s*hd + d) * 4
// ============================================================
.visible .entry transpose_shd_hsd_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_S,
    .param .u32 p_H,
    .param .u32 p_HD
) {
    .reg .u32 %h, %s, %d, %S, %H, %HD, %src_off_u32, %dst_off_u32, %row_stride;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %val;
    .reg .pred %pred;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %S,   [p_S];
    ld.param.u32 %H,   [p_H];
    ld.param.u32 %HD,  [p_HD];

    mov.u32 %h, %ctaid.x;
    mov.u32 %s, %ctaid.y;
    mov.u32 %d, %tid.x;

    setp.ge.u32 %pred, %d, %HD;
    @%pred bra $L_tp_done;
    setp.ge.u32 %pred, %h, %H;
    @%pred bra $L_tp_done;
    setp.ge.u32 %pred, %s, %S;
    @%pred bra $L_tp_done;

    // src_off = s*H*HD + h*HD + d
    mul.lo.u32 %row_stride, %H, %HD;
    mul.lo.u32 %src_off_u32, %s, %row_stride;
    mul.lo.u32 %row_stride, %h, %HD;
    add.u32 %src_off_u32, %src_off_u32, %row_stride;
    add.u32 %src_off_u32, %src_off_u32, %d;
    mul.wide.u32 %off, %src_off_u32, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %val, [%off];

    // dst_off = h*S*HD + s*HD + d
    mul.lo.u32 %row_stride, %S, %HD;
    mul.lo.u32 %dst_off_u32, %h, %row_stride;
    mul.lo.u32 %row_stride, %s, %HD;
    add.u32 %dst_off_u32, %dst_off_u32, %row_stride;
    add.u32 %dst_off_u32, %dst_off_u32, %d;
    mul.wide.u32 %off, %dst_off_u32, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;

$L_tp_done:
    ret;
}

// ============================================================
// softmax_bwd_f32 (A-LLM-1 Stage 4, 2026-07-26):
//   Row-wise softmax backward: dS_ij = P_ij * (dP_ij - sum_k(P_ik * dP_ik)).
//   Grid: (rows, 1, 1). Block: (1, 1, 1). Single-thread per row (reference kernel,
//   NOT a hot path -- used only for F32-reconstruct attention bwd).
// Params: p_P (in), p_dP (in), p_dS (out), p_rows (u32), p_cols (u32).
// ============================================================
.visible .entry softmax_bwd_f32(
    .param .u64 p_P,
    .param .u64 p_dP,
    .param .u64 p_dS,
    .param .u32 p_rows,
    .param .u32 p_cols
) {
    .reg .u64 %P, %dP, %dS, %row_base, %off;
    .reg .u32 %tidx, %bidx, %rows, %cols, %i, %row_off;
    .reg .f32 %p_val, %dp_val, %s_val, %sum_r;
    .reg .pred %pred;

    ld.param.u64 %P,     [p_P];
    ld.param.u64 %dP,    [p_dP];
    ld.param.u64 %dS,    [p_dS];
    ld.param.u32 %rows,  [p_rows];
    ld.param.u32 %cols,  [p_cols];

    mov.u32 %bidx, %ctaid.x;
    setp.ge.u32 %pred, %bidx, %rows;
    @%pred bra $L_sbwd_done;

    mov.u32 %tidx, %tid.x;
    setp.ne.u32 %pred, %tidx, 0;
    @%pred bra $L_sbwd_done;

    // row_base = P + row * cols * 4
    mul.lo.u32 %row_off, %bidx, %cols;
    mul.wide.u32 %row_base, %row_off, 4;

    // Pass 1: sum_r = sum(P * dP) over row.
    mov.f32 %sum_r, 0f00000000;
    mov.u32 %i, 0;
$L_sbwd_p1:
    setp.ge.u32 %pred, %i, %cols;
    @%pred bra $L_sbwd_p1_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %off, %row_base;
    add.u64 %off, %off, %P;
    ld.global.f32 %p_val, [%off];
    sub.u64 %off, %off, %P;
    add.u64 %off, %off, %dP;
    ld.global.f32 %dp_val, [%off];
    fma.rn.f32 %sum_r, %p_val, %dp_val, %sum_r;
    add.u32 %i, %i, 1;
    bra $L_sbwd_p1;
$L_sbwd_p1_end:

    // Pass 2: dS[row, i] = P[row, i] * (dP[row, i] - sum_r)
    mov.u32 %i, 0;
$L_sbwd_p2:
    setp.ge.u32 %pred, %i, %cols;
    @%pred bra $L_sbwd_done;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %off, %row_base;
    add.u64 %off, %off, %P;
    ld.global.f32 %p_val, [%off];
    sub.u64 %off, %off, %P;
    add.u64 %off, %off, %dP;
    ld.global.f32 %dp_val, [%off];
    sub.f32 %s_val, %dp_val, %sum_r;
    mul.f32 %s_val, %s_val, %p_val;
    sub.u64 %off, %off, %dP;
    add.u64 %off, %off, %dS;
    st.global.f32 [%off], %s_val;
    add.u32 %i, %i, 1;
    bra $L_sbwd_p2;

$L_sbwd_done:
    ret;
}
` + "\x00"

// Phase B kernel names
var kernelNames_B = []string{
	"abs_f32",
	"log_f32",
	"sqrt_f32",
	"tanh_f32",
	"relu_f32",
	"gelu_f32",
	"sigmoid_f32",
	"sub_f32",
	"div_f32",
	"sum_reduce_f32",
	"max_reduce_f32",
	"layernorm_f32",
	"arange_f32",
	"where_f32",
	"cross_entropy_f32",       // A-1 (2026-07-24)
	"transpose_shd_hsd_f32",   // A-LLM-1 Stage 3 (2026-07-25)
	"softmax_bwd_f32",         // A-LLM-1 Stage 4 (2026-07-26)
}
