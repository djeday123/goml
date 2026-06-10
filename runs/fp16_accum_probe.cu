#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void k() {
    uint32_t d0=0, d1=0;
    uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
    uint32_t c0=0, c1=0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
int main() { printf("compiled\n"); return 0; }
