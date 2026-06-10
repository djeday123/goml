// sass_reference.cu — reference kernels to extract MMA opcode bit patterns
//
// Build per variant, then cuobjdump -sass --dump-bin to see exact bytes.
// We need to compare:
//   A) FP8 + F32 accumulator → QMMA.16832.F32.E4M3.E4M3
//   B) FP8 + F16 accumulator → QMMA.16832.F16.E4M3.E4M3
//   C) FP4 + F32 accumulator → UTCMMA or whatever Blackwell calls it
// Then guess: FP4 + F16/F8 accumulator → patch C's accumulator-type bits.

#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

#if defined(FP8_F32)
__global__ void mma_kernel(uint32_t *out)
{
    uint32_t a0=0x3C3C3C3Cu, a1=0x3C3C3C3Cu, a2=0x3C3C3C3Cu, a3=0x3C3C3C3Cu;
    uint32_t b0=0x3C3C3C3Cu, b1=0x3C3C3C3Cu;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    if (threadIdx.x == 0) {
        out[0] = (uint32_t)c0; out[1] = (uint32_t)c1;
        out[2] = (uint32_t)c2; out[3] = (uint32_t)c3;
    }
}
#endif

#if defined(FP8_F16)
__global__ void mma_kernel(uint32_t *out)
{
    uint32_t a0=0x3C3C3C3Cu, a1=0x3C3C3C3Cu, a2=0x3C3C3C3Cu, a3=0x3C3C3C3Cu;
    uint32_t b0=0x3C3C3C3Cu, b1=0x3C3C3C3Cu;
    uint32_t c0=0, c1=0;
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1));
    if (threadIdx.x == 0) { out[0] = d0; out[1] = d1; }
}
#endif

#if defined(FP4_F32)
__global__ void mma_kernel(uint32_t *out)
{
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,sa=0,sb=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    if (threadIdx.x == 0) {
        out[0] = (uint32_t)c0; out[1] = (uint32_t)c1;
        out[2] = (uint32_t)c2; out[3] = (uint32_t)c3;
    }
}
#endif

int main() { return 0; }
