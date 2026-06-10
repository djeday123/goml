// hidden_mma_probe.cu — hunt for undocumented mma variants on sm_120a
// Hypothesis: NVIDIA may hide instructions that give 2× speedup
// (like the f16 accum trick on 4090).

#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

// 1. FP8 e4m3 inputs WITH FP8 e4m3 output (accumulator)
#if defined(PROBE_FP8_FP8_ACC)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.e4m3.e4m3.e4m3.e4m3 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
#endif

// 2. FP8 with FP8 accum via kind::f8f6f4
#if defined(PROBE_FP8_FP8_ACC_KIND)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.e4m3.e4m3.e4m3.e4m3 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
#endif

// 3. Sparse mma FP8 — 2:4 structured sparsity, k=64. For sparse:
//    A operand same as dense m16n8k32 (4 uint32) — same nonzeros
//    B operand DOUBLED (4 uint32, not 2) — full k=64
#if defined(PROBE_SP_FP8)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t b0=0,b1=0,b2=0,b3=0;
    uint32_t meta=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12, 0x0;\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));
}
#endif

// 4. Sparse mma FP8 with kind::f8f6f4 (Blackwell unified path)
#if defined(PROBE_SP_FP8_KIND)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    uint32_t meta=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10, 0x0;\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(meta));
}
#endif

// 5. Sparse FP4 (m16n8k128 — 2× FP4 dense k=64)
#if defined(PROBE_SP_FP4)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t meta=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10, 0x0, %11, {0,0}, %12, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(meta), "r"(sa), "r"(sb));
}
#endif

// 6. INT8 sparse — same arg size as sparse FP8 (B doubled)
#if defined(PROBE_SP_INT8)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t b0=0,b1=0,b2=0,b3=0;
    uint32_t meta=0;
    int c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12, 0x0;\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));
}
#endif

// 7. INT4 dense for comparison (legacy on Ada, maybe still on Blackwell)
#if defined(PROBE_INT4)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    int c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

int main() { printf("compiled\n"); return 0; }
