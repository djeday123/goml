// fp4_hidden_probe.cu — hunt for hidden FP4 paths above 2198 TFLOPS
//
// Candidates:
//   1. Sparse FP4 with .sp::ordered_metadata modifier (newer syntax)
//   2. Larger K shapes (m16n8k128) for FP4
//   3. Larger m (m32n8k64) for FP4
//   4. Dense INT4 (lowest precision, untested)
//   5. Sparse INT4
//   6. mxf4 with FP4 output (no upcast)
//   7. Different kind:: modifiers

#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

// 1. Sparse FP4 with ordered_metadata
#if defined(PROBE_SP_FP4_ORDERED)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t b0=0,b1=0,b2=0,b3=0;
    uint32_t meta=0, sa=0, sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12, 0x0, %13, {0,0}, %14, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta), "r"(sa), "r"(sb));
}
#endif

// 2. Larger K FP4 — m16n8k128 (4× of m16n8k32)
#if defined(PROBE_FP4_K128)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    uint32_t b0=0,b1=0,b2=0,b3=0;
    uint32_t sa=0, sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7,%8,%9,%10,%11},{%12,%13,%14,%15},{%0,%1,%2,%3}, %16, {0,0}, %17, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(sa), "r"(sb));
}
#endif

// 3. Dense INT4 m16n8k64 (lowest precision)
#if defined(PROBE_INT4_DENSE)
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

// 4. Sparse INT4 m16n8k128
#if defined(PROBE_INT4_SPARSE)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t b0=0,b1=0,b2=0,b3=0;
    uint32_t meta=0;
    int c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12, 0x0;\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));
}
#endif

// 5. FP4 with .e4m3 output (potential trick: keep result in FP8)
#if defined(PROBE_FP4_FP8_OUT)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;  // 2 uint32 = 4 fp8 packed
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.e4m3.e2m1.e2m1.e4m3.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9}, %10, {0,0}, %11, {0,0};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(sa), "r"(sb));
}
#endif

// 6. Dense FP4 m16n8k32 (smaller — maybe faster per cycle?)
#if defined(PROBE_FP4_K32)
__global__ void k() {
    uint32_t a0=0,a1=0,b0=0;
    uint32_t sa=0, sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3}, %7, {0,0}, %8, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(sa), "r"(sb));
}
#endif

// 7. Wider m for FP4 — m32n8k64
#if defined(PROBE_FP4_M32)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    uint32_t b0=0,b1=0;
    uint32_t sa=0, sb=0;
    float c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;
    asm volatile(
        "mma.sync.aligned.m32n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17},{%0,%1,%2,%3,%4,%5,%6,%7}, %18, {0,0}, %19, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3), "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// 8. FP4 with NVIDIA's nvf4 scale_vec::4X (denser scaling)
#if defined(PROBE_FP4_NVF4_DENSE)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

int main() { printf("compiled\n"); return 0; }
