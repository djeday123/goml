// Layer 1 probe: ldmatrix.m8n8.x4.b16 bank conflict measurement on stride-128.
// Strategy: many blocks × 1 ldmatrix per block (no loop — compiler doesn't fold).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} } while(0)

constexpr int STRIDE = 128;
constexpr int ROWS   = 8;
constexpr int SMEM_BYTES = ROWS * STRIDE;
constexpr int NBLOCKS = 16384;   // match production dQ block count

// Kernel A: 1 ldmatrix.m8n8.x4.b16 per block
__global__ void probe_ldmatrix(const uint8_t* in, uint32_t* out_regs) {
    extern __shared__ uint8_t sm[];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_BYTES; i += blockDim.x) sm[i] = in[i];
    __syncthreads();

    int row = tid & 7;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm[row * STRIDE]));

    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));

    // Write to global to prevent dead-code elimination
    int bid = blockIdx.x;
    out_regs[bid * 128 + tid * 4 + 0] = r0;
    out_regs[bid * 128 + tid * 4 + 1] = r1;
    out_regs[bid * 128 + tid * 4 + 2] = r2;
    out_regs[bid * 128 + tid * 4 + 3] = r3;
}

// Kernel B: 4 LDS per block (mirror dQ MMA-A K read pattern, baseline 8-way conflict)
__global__ void probe_lds(const uint8_t* in, uint32_t* out_regs) {
    extern __shared__ uint8_t sm[];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_BYTES; i += blockDim.x) sm[i] = in[i];
    __syncthreads();

    int lane_k = tid >> 2;
    int lane_n = tid & 3;
    int row = lane_k & 7;
    int col_base = lane_n * 4;

    uint32_t r0 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 0]);
    uint32_t r1 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 16]);
    uint32_t r2 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 32]);
    uint32_t r3 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 48]);

    int bid = blockIdx.x;
    out_regs[bid * 128 + tid * 4 + 0] = r0;
    out_regs[bid * 128 + tid * 4 + 1] = r1;
    out_regs[bid * 128 + tid * 4 + 2] = r2;
    out_regs[bid * 128 + tid * 4 + 3] = r3;
}

int main(int argc, char** argv) {
    uint8_t host_in[SMEM_BYTES];
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < STRIDE; c++)
            host_in[r * STRIDE + c] = static_cast<uint8_t>((r << 4) | (c & 0xf));

    uint8_t  *d_in   = nullptr;
    uint32_t *d_out_lm = nullptr;
    uint32_t *d_out_ld = nullptr;
    CK(cudaMalloc(&d_in, SMEM_BYTES));
    CK(cudaMalloc(&d_out_lm, NBLOCKS * 128 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_out_ld, NBLOCKS * 128 * sizeof(uint32_t)));
    CK(cudaMemcpy(d_in, host_in, SMEM_BYTES, cudaMemcpyHostToDevice));

    probe_ldmatrix<<<NBLOCKS, 32, SMEM_BYTES>>>(d_in, d_out_lm);
    probe_lds<<<NBLOCKS, 32, SMEM_BYTES>>>(d_in, d_out_ld);
    CK(cudaDeviceSynchronize());

    printf("Done. NBLOCKS=%d. NCu measures wavefronts/conflicts aggregated across blocks.\n", NBLOCKS);
    cudaFree(d_in); cudaFree(d_out_lm); cudaFree(d_out_ld);
    return 0;
}
