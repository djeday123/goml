// sm120_capabilities.cu — probe each instruction family individually.
// Compile separately per probe using:
//   nvcc -arch=sm_120 -DPROBE_<NAME> -O3 sm120_capabilities.cu -o probe_<name>
// If ptxas errors → not supported. If it compiles → supported.

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#if defined(PROBE_TMA_BULK)
// TMA via cp.async.bulk.tensor.2d — needs CUtensorMap descriptor + mbarrier
__global__ void k(const __grid_constant__ CUtensorMap tm) {
    __shared__ alignas(128) __half smem[64 * 128];
    __shared__ alignas(8) uint64_t bar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
               "l"((const void*)&tm),
               "r"(0), "r"(0),
               "r"((uint32_t)__cvta_generic_to_shared(&bar)));
    }
}
#endif

#if defined(PROBE_STMATRIX_X4)
// stmatrix.x4 — store 4×(8×8) matrix from regs to SMEM
__global__ void k() {
    __shared__ alignas(16) __half smem[64 * 8];
    uint32_t r0 = 1, r1 = 2, r2 = 3, r3 = 4;
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
                    "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
#endif

#if defined(PROBE_STMATRIX_X4_TRANS)
__global__ void k() {
    __shared__ alignas(16) __half smem[64 * 8];
    uint32_t r0 = 1, r1 = 2, r2 = 3, r3 = 4;
    asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
                    "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
#endif

#if defined(PROBE_MBARRIER)
__global__ void k() {
    __shared__ alignas(8) uint64_t bar;
    asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
    uint32_t st;
    asm volatile("mbarrier.try_wait.parity.shared.b64 %0, [%1], 0;\n"
                 : "=r"(st) : "r"((uint32_t)__cvta_generic_to_shared(&bar)));
}
#endif

#if defined(PROBE_FP8_MMA)
// FP8 mma: m16n8k32 e4m3 from sm_89, available on sm_120 too?
__global__ void k() {
    float d0=0, d1=0, d2=0, d3=0;
    uint32_t a0=0, a1=0, a2=0, a3=0;
    uint32_t b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

#if defined(PROBE_FP6_MMA)
// FP6 mma: m16n8k32 e3m2 — Blackwell new
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0;
    uint32_t b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

#if defined(PROBE_FP4_MMA)
// FP4 mma e2m1: m16n8k64? — Blackwell
__global__ void k() {
    float d0=0;
    uint32_t a0=0, a1=0, a2=0, a3=0;
    uint32_t b0=0, b1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32 "
        "{%0,%0,%0,%0},{%1,%2,%3,%4},{%5,%6},{%0,%0,%0,%0};\n"
        : "+f"(d0)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

int main() {
    cudaError_t e = cudaGetLastError();
    printf("compiled OK, last cudaError=%d\n", e);
    return 0;
}
