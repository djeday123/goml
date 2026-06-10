// fp4_fp8acc_v2.cu — exhaustive search for FP4 inputs + FP8 accumulator path

#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

// 1. kind::mxf4 + e4m3 D, .f32 C (mixed: f32 input accum, f8 output)
#if defined(PROBE_F32C_E4M3D)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.e4m3.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9,%10,%11}, %12, {0,0}, %13, {0,0};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
}
#endif

// 2. kind::mxf4 + .f16 accum
#if defined(PROBE_F16_ACC_V2)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f16.e2m1.e2m1.f16.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9}, %10, {0,0}, %11, {0,0};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(sa), "r"(sb));
}
#endif

// 3. kind::mxf4nvf4 + f16 accum
#if defined(PROBE_NVF4_F16)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f16.e2m1.e2m1.f16.ue4m3 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9}, %10, {0,0}, %11, {0,0};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(sa), "r"(sb));
}
#endif

// 4. kind::f8f6f4 with e2m1 inputs — different from mxf4
#if defined(PROBE_F8F6F4_E2M1)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// 5. kind::f8f6f4 with FP4 input via k=64 shape (non-mxf4)
#if defined(PROBE_F8F6F4_K64)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// 6. NO scale variant — pure FP4 without block_scale modifier
#if defined(PROBE_FP4_NO_SCALE)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.f32.e2m1.e2m1.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// 7. e4m3 (FP8) input but in FP4-shaped instruction (k=64)
#if defined(PROBE_FP8_K64)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

int main() { printf("compiled\n"); return 0; }
