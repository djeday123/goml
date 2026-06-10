// fp4_fp8acc_probe.cu — Can we use FP8 (e4m3) accumulator with FP4 inputs?
// Standard mma m16n8k64 .kind::mxf4 docs only show .f32 / .f16 accumulator.
// Question: does ptxas accept .e4m3 / .e5m2 accumulator on sm_120a?

#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

// Variant 1: mxf4 with FP16 accum (known to work per our earlier probes)
#if defined(PROBE_F16_ACC)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f16.e2m1.e2m1.f16.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1}, %8, {0,0}, %9, {0,0};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// Variant 2: mxf4 with FP8 e4m3 accum (the question!)
#if defined(PROBE_E4M3_ACC)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.e4m3.e2m1.e2m1.e4m3.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1}, %8, {0,0}, %9, {0,0};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// Variant 3: mxf4 with FP8 e5m2 accum (wider range)
#if defined(PROBE_E5M2_ACC)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.e5m2.e2m1.e2m1.e5m2.ue8m0 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1}, %8, {0,0}, %9, {0,0};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

// Variant 4: mxf4nvf4 with FP8 accum
#if defined(PROBE_NVF4_E4M3_ACC)
__global__ void k() {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.e4m3.e2m1.e2m1.e4m3.ue4m3 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1}, %8, {0,0}, %9, {0,0};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

int main() { printf("compiled\n"); return 0; }
