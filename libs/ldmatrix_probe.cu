// ldmatrix availability probe for sm_120a (Blackwell consumer)
// Find which ldmatrix variants compile. cvta fixed via __cvta_generic_to_shared.

#include <cstdint>
#include <cuda_runtime.h>

// Test 1: ldmatrix b16 (standard) — Ampere+
__global__ void test_b16(uint16_t* sm_in, uint32_t* out) {
    extern __shared__ uint8_t smem[];
    uint16_t* sm = reinterpret_cast<uint16_t*>(smem);
    int tid = threadIdx.x;
    sm[tid] = sm_in[tid];
    __syncthreads();
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm[(tid & 31) * 8]));
    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    if (tid < 4) { out[tid*4]=r0; out[tid*4+1]=r1; out[tid*4+2]=r2; out[tid*4+3]=r3; }
}

// Test 2: ldmatrix b16.trans (transposed)
__global__ void test_b16_trans(uint16_t* sm_in, uint32_t* out) {
    extern __shared__ uint8_t smem[];
    uint16_t* sm = reinterpret_cast<uint16_t*>(smem);
    int tid = threadIdx.x;
    sm[tid] = sm_in[tid];
    __syncthreads();
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm[(tid & 31) * 8]));
    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
    if (tid < 4) { out[tid*4]=r0; out[tid*4+1]=r1; out[tid*4+2]=r2; out[tid*4+3]=r3; }
}

// Test 3: m16n16.b8.trans (Blackwell wider variant for FP8?)
// per the m16n16 error: requires .trans, output vector size 2
__global__ void test_m16n16_b8_trans(uint8_t* sm_in, uint32_t* out) {
    extern __shared__ uint8_t smem[];
    uint8_t* sm = smem;
    int tid = threadIdx.x;
    sm[tid] = sm_in[tid];
    __syncthreads();
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm[(tid & 31) * 8]));
    uint32_t r0, r1;
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(addr));
    if (tid < 4) { out[tid*2]=r0; out[tid*2+1]=r1; }
}

int main() { return 0; }
