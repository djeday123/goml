// =============================================================================
// Bank Conflict Diagnostic
// =============================================================================
// Runs v2-DB and v2-DB-SW once each on a representative workload,
// then you profile with:
//
//   ncu --set full --kernel-name flash_attention \
//       ./runs/flash_bc_diag
//
// Or targeted:
//   ncu --metrics \
//       l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
//       l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second,\
//       smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct,\
//       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
//       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
//       ./runs/flash_bc_diag
//
// Build:
//   nvcc -O3 -arch=sm_89 -lineinfo -std=c++17 \
//     libs/flash_attention_v2_dbsw.cu libs/flash_attention_v2_db.cu \
//     libs/flash_attention_v2.cu libs/flash_attention.cu \
//     libs/transformer_kernels.cu libs/flash_bc_diag.cu \
//     -o runs/flash_bc_diag -lcudart
// =============================================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
{
    int flash_attention_v2_db_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);

    int flash_attention_v2_dbsw_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim,
        int causal, void *stream);
}

#define CK(c)                                                        \
    do                                                               \
    {                                                                \
        cudaError_t e = (c);                                         \
        if (e)                                                       \
        {                                                            \
            printf("ERR %d: %s\n", __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                 \
        }                                                            \
    } while (0)

int main()
{
    // 32 heads, seq=2048, d=128 — representative 7B workload
    int heads = 32, seq = 2048, dim = 128;
    size_t n = (size_t)heads * seq * dim;
    size_t bytes = n * sizeof(__half);

    __half *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, bytes));
    CK(cudaMalloc(&dK, bytes));
    CK(cudaMalloc(&dV, bytes));
    CK(cudaMalloc(&dO, bytes));
    CK(cudaMemset(dQ, 0x3C, bytes));
    CK(cudaMemset(dK, 0x3C, bytes));
    CK(cudaMemset(dV, 0x3C, bytes));

    printf("Running v2-DB (no swizzle)...\n");
    CK(cudaMemset(dO, 0, bytes));
    flash_attention_v2_db_forward(dQ, dK, dV, dO, heads, seq, dim, 1, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    printf("Running v2-DB-SW (swizzle)...\n");
    CK(cudaMemset(dO, 0, bytes));
    flash_attention_v2_dbsw_forward(dQ, dK, dV, dO, heads, seq, dim, 1, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    printf("Done. Profile with ncu to see bank conflicts.\n");

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    return 0;
}
