// Probe all FP8 mma variants on sm_120a to find wider-N alternative to m16n8k32
#include <cstdint>
#include <cuda_runtime.h>

// Test A: m16n8k64 FP8 (double k)?
__global__ void test_m16n8k64(uint32_t* out) {
    uint32_t a0=1,a1=1,a2=1,a3=1,a4=1,a5=1,a6=1,a7=1;
    uint32_t b0=1,b1=1,b2=1,b3=1;
    float d0=0,d1=0,d2=0,d3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19};"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),"r"(a6),"r"(a7),
          "r"(b0),"r"(b1),"r"(b2),"r"(b3),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));
    if (threadIdx.x == 0) out[0] = *(uint32_t*)&d0;
}
int main() { return 0; }
