// Probe: does sm_120a have mma.sync.m16n16k32 for FP8 (e4m3)?
// If YES → path B (switch MMA-C shape + integrate ldmatrix.m16n16.b8.trans)
// If NO  → path B killed → only paths A/C remain

#include <cstdint>
#include <cuda_runtime.h>

// Test 1: mma.sync.m16n16k32.row.col.f32.e4m3.e4m3.f32 — Blackwell wider FP8 MMA?
__global__ void test_m16n16k32_e4m3_f32(uint32_t* out) {
    uint32_t a0=1, a1=1, a2=1, a3=1;
    uint32_t b0=1, b1=1, b2=1, b3=1;
    float d0=0, d1=0, d2=0, d3=0, d4=0, d5=0, d6=0, d7=0;
    asm volatile(
        "mma.sync.aligned.m16n16k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    if (threadIdx.x == 0) out[0] = *reinterpret_cast<uint32_t*>(&d0);
}

int main() { return 0; }
