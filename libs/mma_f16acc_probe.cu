// AA0.1: Verify mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 compiles on sm_120a.
// This is the FP16-accumulator variant of MMA-C (dS * K -> dQ_acc).
// C-fragment: 2 uint32 (holds 4x f16) instead of 4 float — regs 64 -> 32.
#include <cstdint>
#include <cuda_runtime.h>

__global__ void probe_m16n8k32_f16(uint32_t *out) {
    uint32_t a0 = 1, a1 = 2, a2 = 3, a3 = 4;
    uint32_t b0 = 5, b1 = 6;
    uint32_t d0 = 0, d1 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"((uint32_t)0), "r"((uint32_t)0));
    if (threadIdx.x == 0) { out[0] = d0; out[1] = d1; }
}

int main() { return 0; }
