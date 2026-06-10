// mma_shapes_probe.cu — try all known mma.sync shapes for FP16 on sm_120a.
// Currently supported on sm_89: m16n8k16, m16n8k8.
// Hopper added: m64n8k16, m64n16k16, m64n32k16, ... but those are wgmma.
// Question: does sm_120 add a larger mma.sync (non-wgmma) shape?

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#if defined(PROBE_M16N8K8)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, b0=0;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%0,%0,%0},{%1,%2},{%3},{%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(b0));
}
#endif

#if defined(PROBE_M16N8K16)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

#if defined(PROBE_M16N16K16)
__global__ void k() {
    float d0=0, d1=0, d2=0, d3=0, d4=0, d5=0, d6=0, d7=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0, b2=0, b3=0;
    asm volatile(
        "mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
        "{%0,%0,%0,%0,%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6,%7,%8},{%0,%0,%0,%0,%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3));
}
#endif

#if defined(PROBE_M32N8K16)
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0, b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m32n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%0,%0,%0,%0,%0,%0,%0},{%1,%2,%3,%4,%5,%6,%7,%8},{%9,%10},{%0,%0,%0,%0,%0,%0,%0,%0};\n"
        : "+f"(d0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b0), "r"(b1));
}
#endif

int main() {
    printf("probe compiled\n");
    return 0;
}
