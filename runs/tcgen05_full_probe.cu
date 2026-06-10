// tcgen05_full_probe.cu — definitive ISA-level probe for tcgen05 on sm_120a.
//
// Tests each individual instruction of the tcgen05 family with correct PTX
// syntax. If ptxas says "not supported on .target 'sm_120a'", that's the
// definitive ISA answer (per user-recommended Level-1 methodology).
//
// Build each probe in isolation:
//   nvcc -gencode arch=compute_120a,code=sm_120a -DPROBE_<X> ...
//
// Reference: PTX ISA 9.0, sec 9.7.16 (tcgen05) — these are Blackwell DC
// instructions (sm_100a). Whether they exist on consumer sm_120a is the
// question.

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#if defined(PROBE_ALLOC)
__global__ void k() {
    __shared__ alignas(16) uint32_t tmem_addr;
    // Allocate 32 columns in tensor memory; the resulting TMEM address is
    // written to the SMEM destination operand.
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_addr)));
}
#endif

#if defined(PROBE_DEALLOC)
__global__ void k() {
    uint32_t tmem_addr = 0;
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
        :: "r"(tmem_addr));
}
#endif

#if defined(PROBE_FENCE)
__global__ void k() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}
#endif

#if defined(PROBE_COMMIT)
__global__ void k() {
    __shared__ alignas(8) uint64_t mbar;
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
}
#endif

#if defined(PROBE_MMA)
// tcgen05.mma — the main matrix-multiply instruction.
//   d (in TMEM)  — accumulator address
//   a (in SMEM or TMEM) — descriptor
//   b (in SMEM)         — descriptor
//   idesc (in regs)     — instruction descriptor (shape + type)
//   p (1-bit predicate) — accumulate (1) or overwrite (0)
__global__ void k() {
    __shared__ __half smA[64*16];
    __shared__ __half smB[16*8];
    uint32_t d_tmem = 0;
    uint64_t descA = 0, descB = 0;
    uint32_t idesc = 0;
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
        :: "r"(d_tmem), "l"(descA), "l"(descB), "r"(idesc));
}
#endif

#if defined(PROBE_LD)
__global__ void k() {
    uint32_t r0, r1;
    uint32_t tmem_addr = 0;
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(tmem_addr));
}
#endif

#if defined(PROBE_ST)
__global__ void k() {
    uint32_t r0 = 0, r1 = 0;
    uint32_t tmem_addr = 0;
    asm volatile(
        "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2};\n"
        :: "r"(tmem_addr), "r"(r0), "r"(r1));
}
#endif

#if defined(PROBE_WAIT)
__global__ void k() {
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    asm volatile("tcgen05.wait::st.sync.aligned;\n");
}
#endif

int main() { printf("compiled\n"); return 0; }
