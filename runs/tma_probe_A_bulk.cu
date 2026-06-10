// tma_probe_A_bulk.cu — [A] cp.async.bulk + mbarrier on sm_120a
// Isolated test: just the sm_90 bulk-copy primitive with mbarrier signaling.
// If this fails to compile → ptxas rejects the instruction family on sm_120a.
// If this fails to run    → SM has no execution unit for it (firmware/HW gap).
// If this passes          → the bulk primitive is the building block for [B].
#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void k(uint8_t *g)
{
    __shared__ alignas(128) uint8_t smem[1024];
    __shared__ alignas(8) uint64_t bar;

    if (threadIdx.x == 0) {
        uint32_t bar_s  = __cvta_generic_to_shared(&bar);
        uint32_t smem_s = __cvta_generic_to_shared(smem);

        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_s));
        // expect_tx BEFORE bulk so the tx-counter is armed when tx-events fire
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 1024;" :: "r"(bar_s));
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1], 1024, [%2];"
            :: "r"(smem_s), "l"(g), "r"(bar_s));
        // Spin until barrier fires (parity 0 since fresh init).
        asm volatile(
            "{ .reg .pred P;\n"
            "  WAITA: mbarrier.try_wait.parity.shared.b64 P, [%0], 0;\n"
            "  @P bra DONEA;\n"
            "  bra WAITA;\n"
            "  DONEA: }\n"
            :: "r"(bar_s));
        g[0] = smem[0];  // prevent DCE
    }
}

int main()
{
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("[A] cp.async.bulk + mbarrier on %s sm_%d%d\n", p.name, p.major, p.minor);
    uint8_t *d; cudaMalloc(&d, 4096); cudaMemset(d, 0xAB, 4096);
    k<<<1, 32>>>(d);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) printf("[A] RESULT: PASS — bulk + mbarrier work on sm_120a\n");
    else                    printf("[A] RESULT: FAIL — %s\n", cudaGetErrorString(err));
    cudaFree(d);
    return (err == cudaSuccess) ? 0 : 1;
}
