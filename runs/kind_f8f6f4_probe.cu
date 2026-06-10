// kind_f8f6f4_probe.cu — comprehensive probe of mma.sync .kind::f8f6f4
// variants on sm_120a. Covers FP8 (e4m3, e5m2), FP6 (e3m2, e2m3), FP4
// (e2m1) inputs plus block-scaled MX format (.kind::mxf8f6f4).
//
// Build per probe:
//   nvcc -gencode arch=compute_120a,code=sm_120a -DPROBE_<X> ...

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ─────────── kind::f8f6f4, FP8 inputs (e4m3) ───────────
#if defined(PROBE_F8_E4M3)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// ─────────── kind::f8f6f4, FP8 inputs (e5m2) ───────────
#if defined(PROBE_F8_E5M2)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// ─────────── kind::f8f6f4, FP6 inputs (e3m2) ───────────
#if defined(PROBE_F6_E3M2)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// ─────────── kind::f8f6f4, FP6 inputs (e2m3) ───────────
#if defined(PROBE_F6_E2M3)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// ─────────── kind::f8f6f4, FP4 inputs (e2m1) at m16n8k64 ───────────
#if defined(PROBE_F4_E2M1)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// ─────────── kind::mxf8f6f4 — MX block-scaled FP8 (e4m3 with ue8m0 scale) ───────────
#if defined(PROBE_MXF8_E4M3)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t sa=0, sb=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0}, %7, {0,0}, %8, {0,0};\n"
        : "+f"(d0)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// ─────────── kind::mxf4 — MX block-scaled FP4 (e2m1 with ue8m0 scale, m16n8k64) ───────────
#if defined(PROBE_MXF4_E2M1)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t sa=0, sb=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0}, %7, {0,0}, %8, {0,0};\n"
        : "+f"(d0)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// ─────────── kind::mxf4nvf4 — FP4 with FP8 scale (ue4m3) ───────────
#if defined(PROBE_MXF4NVF4_E2M1)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t sa=0, sb=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0}, %7, {0,0}, %8, {0,0};\n"
        : "+f"(d0)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

int main() { printf("compiled\n"); return 0; }
