// ptx_exhaustive_probe.cu — exhaustive search for undocumented PTX modifier
// combinations that might unlock FP4 + FP8 accumulator or other hidden paths.
//
// Strategy: try variations no one would document — typos in modifier order,
// unusual type combos, undocumented kind:: names, etc.
//
// Each probe compiles independently. If ptxas accepts → potential new path.

#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

// === A. Modifier order variants ===

// A1: kind:: BEFORE row.col
#if defined(PROBE_A1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4.m16n8k64.row.col.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// === B. Type name variants ===

// B1: .e4m3fn (some HW docs use the "fn" suffix for accum)
#if defined(PROBE_B1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.e4m3fn.e2m1.e2m1.e4m3fn.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9}, %10, {0,0}, %11, {0,0};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(sa), "r"(sb));
}
#endif

// B2: .f8 (generic, no e4m3/e5m2 distinction)
#if defined(PROBE_B2)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f8.e2m1.e2m1.f8.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9}, %10, {0,0}, %11, {0,0};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(sa), "r"(sb));
}
#endif

// B3: tf32 accumulator (FP32 in TF32 mode)
#if defined(PROBE_B3)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.tf32.e2m1.e2m1.tf32.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9}, %10, {0,0}, %11, {0,0};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(sa), "r"(sb));
}
#endif

// === C. Undocumented kind:: variants ===

// C1: kind::mxf8
#if defined(PROBE_C1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf8.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// C2: kind::mxf6 (FP6 specific)
#if defined(PROBE_C2)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf6.block_scale.scale_vec::2X.f32.e3m2.e3m2.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// C3: kind::f8 (just FP8 family)
#if defined(PROBE_C3)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// C4: kind::f4 (just FP4)
#if defined(PROBE_C4)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::f4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// === D. Scale variants ===

// D1: scale_vec::1X
#if defined(PROBE_D1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// D2: no block_scale but with scale_vec
#if defined(PROBE_D2)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// D3: ue3m2 (FP6 scale)
#if defined(PROBE_D3)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue3m2 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// === E. Mixed input types ===

// E1: e2m1 × e4m3 (FP4 × FP8)
#if defined(PROBE_E1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// === F. SP modifiers ===

// F1: sp::unordered_metadata (alternative form)
#if defined(PROBE_F1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t b0=0,b1=0,b2=0,b3=0;
    uint32_t meta=0, sa=0, sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sp::unordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12, 0x0, %13, {0,0}, %14, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta), "r"(sa), "r"(sb));
}
#endif

// === G. Larger accumulators with FP4 ===

// G1: Output 2x as wide (8 fp32 instead of 4) — m32 alternative
#if defined(PROBE_G1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;
    asm volatile(
        "mma.sync.aligned.m16n16k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13},{%0,%1,%2,%3,%4,%5,%6,%7}, %14, {0,0}, %15, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3), "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

int main() { printf("compiled\n"); return 0; }
