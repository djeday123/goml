// =====================================================================
//  probe_cpasync_transpose.cu — empirical verification of cp.async
//  "contiguous-to-contiguous" claim. Tests if cp.async can fill smem
//  in COL-MAJOR Q_T layout from ROW-MAJOR Q in gmem.
//
//  Paper claim: NO. cp.async semantics force contiguous dest bytes from
//  contiguous source bytes. Stride-68 (or anything) doesn't change this.
//  Empirical: try all sizes (.4 ca, .8 ca, .16 cg), measure what
//  actually ends up in smem.
//
//  Binary result:
//    smem matches expected col-major Q_T → spec interpretation WRONG,
//      cp.async transposes somehow → LIVE lever, escalate.
//    smem garbage / scrambled / row-major → spec confirmed → DEAD lever,
//      paper debt closed by measurement.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define BR 64
#define HD 128
#define THREADS 128

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

__device__ __forceinline__ void cpa_4(void *s, const void *g) {
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;"
                 ::"r"(sa), "l"(g));
}
__device__ __forceinline__ void cpa_8(void *s, const void *g) {
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0],[%1],8;"
                 ::"r"(sa), "l"(g));
}
__device__ __forceinline__ void cpa_16(void *s, const void *g) {
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;"
                 ::"r"(sa), "l"(g));
}
__device__ __forceinline__ void cpa_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cpa_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// =====================================================================
// Probe kernel — selects one of 4 variants:
//   V=0: row-major control (standard usage, expected to fill row-major)
//   V=1: col-major attempt via 4-byte cp.async
//   V=2: col-major attempt via 8-byte cp.async
//   V=3: col-major attempt via 16-byte cp.async
//
// "Col-major attempt" interpretation: per-thread issues cp.async with
//   src = gmem[Q row i, cols d..d+N-1]  (N contiguous gmem bytes from row i of Q)
//   dst = smem[smQ_T row d, col i]      (positioned as if dest were col-major Q_T)
// If cp.async honors contiguous semantics, the 4/8/16 src bytes go to
// CONTIGUOUS dst bytes starting at smem[d*BR+i] — landing in smQ_T row d,
// cols i..i+N-1, holding Q[i, d..d+N-1] values. That's WRONG for col-major.
// Only smem[d*BR+i] gets the "right" byte (Q[i, d]) — rest 3/7/15 wrong.
// =====================================================================
__global__ void probe_kernel(
    const uint8_t * __restrict__ Q_g,
    uint8_t       * __restrict__ smem_dump,
    int variant)
{
    extern __shared__ uint8_t smem[];
    const int tid = threadIdx.x;

    // Pre-zero smem to detect un-written bytes
    for (int e = tid; e < BR * HD; e += THREADS) smem[e] = 0xAA;
    __syncthreads();

    if (variant == 0) {
        // V0: row-major control — copy whole Q to smem row-major using cp.async.16
        constexpr int CHUNK = 16;
        for (int e = tid; e < BR * HD / CHUNK; e += THREADS) {
            int gmem_off = e * CHUNK;
            int smem_off = e * CHUNK;
            cpa_16(&smem[smem_off], &Q_g[gmem_off]);
        }
    }
    else if (variant == 1) {
        // V1: col-major attempt, cp.async.4 — gated by 4-byte alignment of dst
        constexpr int CHUNK = 4;
        for (int e = tid; e < BR * (HD / CHUNK); e += THREADS) {
            int i       = e / (HD / CHUNK);
            int d_chunk = e % (HD / CHUNK);
            int d_base  = d_chunk * CHUNK;
            int gmem_off = i * HD + d_base;
            int smem_off = d_base * BR + i;
            if ((smem_off & 3) == 0) {
                cpa_4(&smem[smem_off], &Q_g[gmem_off]);
            }
        }
    }
    else if (variant == 2) {
        // V2: col-major attempt, cp.async.8 — gated by 8-byte alignment
        constexpr int CHUNK = 8;
        for (int e = tid; e < BR * (HD / CHUNK); e += THREADS) {
            int i       = e / (HD / CHUNK);
            int d_chunk = e % (HD / CHUNK);
            int d_base  = d_chunk * CHUNK;
            int gmem_off = i * HD + d_base;
            int smem_off = d_base * BR + i;
            if ((smem_off & 7) == 0) {
                cpa_8(&smem[smem_off], &Q_g[gmem_off]);
            }
        }
    }
    else if (variant == 3) {
        // V3: col-major attempt, cp.async.16 — gated by 16-byte alignment
        constexpr int CHUNK = 16;
        for (int e = tid; e < BR * (HD / CHUNK); e += THREADS) {
            int i       = e / (HD / CHUNK);
            int d_chunk = e % (HD / CHUNK);
            int d_base  = d_chunk * CHUNK;
            int gmem_off = i * HD + d_base;
            int smem_off = d_base * BR + i;
            if ((smem_off & 15) == 0) {
                cpa_16(&smem[smem_off], &Q_g[gmem_off]);
            }
        }
    }

    cpa_commit();
    cpa_wait_all();
    __syncthreads();

    // Dump smem to gmem for host inspection
    for (int e = tid; e < BR * HD; e += THREADS) {
        smem_dump[e] = smem[e];
    }
}

int main()
{
    // Deterministic Q: Q[i, d] = (i*HD + d) & 0xFF, so byte content identifies (i, d).
    uint8_t Q_h[BR * HD];
    for (int i = 0; i < BR; ++i)
        for (int d = 0; d < HD; ++d)
            Q_h[i * HD + d] = (uint8_t)((i * HD + d) & 0xFF);

    // Expected col-major Q_T: Q_T[d, i] = Q[i, d]
    uint8_t Q_T_exp[BR * HD];
    for (int d = 0; d < HD; ++d)
        for (int i = 0; i < BR; ++i)
            Q_T_exp[d * BR + i] = Q_h[i * HD + d];

    uint8_t *Q_d, *dump_d;
    CK(cudaMalloc(&Q_d, BR * HD));
    CK(cudaMalloc(&dump_d, BR * HD));
    CK(cudaMemcpy(Q_d, Q_h, BR * HD, cudaMemcpyHostToDevice));

    CK(cudaFuncSetAttribute(probe_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            BR * HD));

    const char *names[] = {
        "V0 row-major control (cp.async.16)",
        "V1 col-major attempt (cp.async.4)",
        "V2 col-major attempt (cp.async.8)",
        "V3 col-major attempt (cp.async.16)"
    };

    for (int v = 0; v < 4; ++v) {
        CK(cudaMemset(dump_d, 0x55, BR * HD));
        probe_kernel<<<1, THREADS, BR * HD>>>(Q_d, dump_d, v);
        CK(cudaDeviceSynchronize());

        uint8_t smem_h[BR * HD];
        CK(cudaMemcpy(smem_h, dump_d, BR * HD, cudaMemcpyDeviceToHost));

        int match_col_major = 0;
        int match_row_major = 0;
        int unwritten = 0;       // 0xAA preserved → not written
        for (int k = 0; k < BR * HD; ++k) {
            if (smem_h[k] == Q_T_exp[k]) match_col_major++;
            if (smem_h[k] == Q_h[k])     match_row_major++;
            if (smem_h[k] == 0xAA)       unwritten++;
        }

        printf("%s\n", names[v]);
        printf("  match col-major Q_T: %d/%d (%.1f%%)\n",
               match_col_major, BR * HD, 100.0 * match_col_major / (BR * HD));
        printf("  match row-major Q:   %d/%d (%.1f%%)\n",
               match_row_major, BR * HD, 100.0 * match_row_major / (BR * HD));
        printf("  unwritten (0xAA):    %d/%d (%.1f%%)\n",
               unwritten, BR * HD, 100.0 * unwritten / (BR * HD));

        // First 16 bytes of smem (vs expected col-major)
        printf("  smem[0..15]:      ");
        for (int k = 0; k < 16; ++k) printf("%02x ", smem_h[k]);
        printf("\n");
        printf("  col-major exp:    ");
        for (int k = 0; k < 16; ++k) printf("%02x ", Q_T_exp[k]);
        printf("\n");
        printf("  row-major Q:      ");
        for (int k = 0; k < 16; ++k) printf("%02x ", Q_h[k]);
        printf("\n\n");
    }

    CK(cudaFree(Q_d));
    CK(cudaFree(dump_d));
    return 0;
}
