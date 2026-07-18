// TMA (Tensor Memory Accelerator) PTX-level availability probe for sm_120a.
// Tests: cp.async.bulk / cp.async.bulk.tensor PTX instructions compile on compute_120a.
// If nvcc errors "not supported" — TMA absent on sm_120a. If compiles + real SASS — TMA present.

#include <cstdint>
#include <cuda_runtime.h>

// Test 1: cp.async.bulk.tensor.2d.shared::cluster.global — TMA tensor load (needs tensor_map)
__global__ void test_tma_tensor_2d(const void* tensor_map, uint8_t* out) {
    __shared__ __align__(128) uint8_t sm[128 * 128];
    __shared__ __align__(8)  uint64_t mbar[1];
    int tid = threadIdx.x;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"((uint32_t)__cvta_generic_to_shared(sm)),
               "l"(tensor_map),
               "r"(0), "r"(0),
               "r"((uint32_t)__cvta_generic_to_shared(mbar)));
    }
    __syncthreads();
    if (tid < 128*128) out[tid] = sm[tid];
}

// Test 2: cp.async.bulk (raw, no tensor map, no swizzle) — simpler variant
__global__ void test_bulk_raw(const uint8_t* g, uint8_t* out) {
    __shared__ __align__(128) uint8_t sm[1024];
    __shared__ __align__(8)  uint64_t mbar[1];
    int tid = threadIdx.x;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1], %2, [%3];"
            :: "r"((uint32_t)__cvta_generic_to_shared(sm)),
               "l"(g),
               "n"(1024),
               "r"((uint32_t)__cvta_generic_to_shared(mbar)));
    }
    __syncthreads();
    if (tid < 1024) out[tid] = sm[tid];
}

int main() { return 0; }
