// kind_f8f6f4_f16acc_probe.cu — does kind::f8f6f4 accept f16 accumulator?
// PTX docs say only .f32 but we found on 4090 (sm_89) that the f16 form
// of the OLD mma path exists even though it wasn't documented in spec.
// Re-test the new family.
//
// Permutations to try (output_type . input_a . input_b . accum_type):
//   f32.e4m3.e4m3.f32   — known to compile (just verified)
//   f16.e4m3.e4m3.f16   — the variant we want
//   f16.e4m3.e4m3.f32   — mixed: f16 out, f32 accum
//   f32.e4m3.e4m3.f16   — mixed: f32 out, f16 accum
//   f16.e5m2.e5m2.f16   — e5m2 variant
//
// Build:
//   nvcc -gencode arch=compute_120a,code=sm_120a -DPROBE_X -O3 ...

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#if defined(PROBE_F16_F16)
// kind::f8f6f4 with f16 output, f16 accum — the "do we keep v23's accum?" probe
__global__ void k() {
    uint32_t d0=0, d1=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
#endif

#if defined(PROBE_F16_F32)
// f16 output, f32 accum — mixed
__global__ void k() {
    uint32_t d0=0, d1=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f32 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9,%10,%11};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
#endif

#if defined(PROBE_F32_F16)
// f32 output, f16 accum — mixed
__global__ void k() {
    float d0=0, d1=0, d2=0, d3=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f16 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
#endif

#if defined(PROBE_F16_F16_E5M2)
__global__ void k() {
    uint32_t d0=0, d1=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e5m2.e5m2.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
#endif

#if defined(PROBE_F16_F16_E3M2)
// FP6 e3m2 with f16 accum?
__global__ void k() {
    uint32_t d0=0, d1=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e3m2.e3m2.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
#endif

int main() { printf("compiled\n"); return 0; }
