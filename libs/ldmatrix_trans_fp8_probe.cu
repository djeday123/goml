// =============================================================================
// ldmatrix.trans FP8 B-operand probe (standalone)
// =============================================================================
// Isolated falsifiable test: does ldmatrix.trans correctly load FP8 B-fragment
// for mma.sync.aligned.m16n8k32.kind::f8f6f4 from K-major smV layout?
//
// Test setup:
//   1. Initialize smV with known pattern: smV[k][n] = k * 256 + n (8-bit, mod 256)
//   2. Use ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 to load B-fragment
//   3. Each thread writes its 2 uint32 (= 8 FP8 bytes) to gmem
//   4. CPU verifies: does this match the expected transposed N-major layout?
//
// Per PV MMA m16n8k32 B-fragment:
//   B is [n8, k32] = 8 rows × 32 cols of FP8 = 256 bytes total per fragment
//   Spread across 32 threads → 8 bytes / thread = 2 uint32
//
// ldmatrix.x2.trans.b16:
//   Loads 2 8x8 b16 matrices from SMEM (transposed)
//   Total: 2 * 128 bytes = 256 bytes, 8 bytes / thread = 2 uint32 ✓
//
// If passes → drop smV_T (8.5 KB freed) → enables 5-stage buffer.
// If fails → diagnose layout (which elements mismatched) → tune.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <cuda_runtime.h>

#define BC 32  // K-tile size (matches m16n8k32 k=32)
#define N  64  // N-tile size, only first 8 used by single B-fragment
#define V_STRIDE 64  // smV stride (BC × N bytes per row in K-major)

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

// Load 8x8 b16 matrices from shared memory with transposition.
// x2 = 2 matrices loaded.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1, const void *smem_ptr)
{
    uint32_t sa;
    asm("cvta.to.shared.u64 %0, %1;" : "=r"(sa) : "l"(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(sa)
    );
}

__global__ void ldmatrix_trans_probe_kernel(uint32_t *out)
{
    __shared__ uint8_t smV[BC * N];  // K-major: smV[k][n], k in [0,BC), n in [0,N)
    int tid = threadIdx.x;

    // ===== Init smV with known pattern =====
    // Each thread loads 8 consecutive bytes (smV total = BC*N = 32*64 = 2048 bytes, 64 bytes/thread for 32 threads).
    // We use a simpler init: known pattern smV[k][n] = (k * 13 + n * 7) mod 256.
    constexpr int TOTAL_BYTES = BC * N;
    constexpr int BYTES_PER_THREAD = TOTAL_BYTES / 32;
    for (int b = 0; b < BYTES_PER_THREAD; b++) {
        int idx = tid * BYTES_PER_THREAD + b;
        int k = idx / N;
        int n = idx % N;
        smV[idx] = (uint8_t)((k * 13 + n * 7) & 0xFF);
    }
    __syncthreads();

    // ===== Apply ldmatrix.x2.trans on first 8x16 region (k=0..7, n=0..15 in b16 = 0..31 in FP8) =====
    // The trans variant transposes: input [k=8, n=8] b16 → output transposed [n=8, k=8] b16.
    // Address pointer: thread (lane) within group of 8 maps to a row offset in smV.
    // For ldmatrix.m8n8: thread t in lanes 0..7 each provides smem_ptr for row t (within 8 rows).
    //   Threads 0..7 → rows 0..7 of smV
    //   Threads 8..15 → second matrix's rows 0..7
    // For x2 = 2 matrices.
    // Per PTX docs, smem_ptr operand from threads 0..15 is used; threads 16..31 are unused for address.

    // Compute address: thread t gets pointer to smV[t % 8][...]
    // For x2.trans: each thread provides smem_ptr to row (t & 7); both matrices arrive in r0 (matrix 0) and r1 (matrix 1).
    int row_in_matrix = tid & 7;       // 0..7 for the 8x8 layout
    int mat_idx = (tid >> 3) & 1;       // 0 or 1: which of x2 matrices
    // Matrix 0: rows from smV[k=0..7][n=0..15 bytes = 0..15 b16 with 8 elements]
    // Matrix 1: rows from smV[k=0..7][n=16..31 bytes = 8..15 b16 with 8 elements]
    // Each row of source has 16 bytes (= 8 b16 elements).
    int src_offset = row_in_matrix * V_STRIDE + mat_idx * 16;  // 16 bytes per matrix column band
    const void *smem_ptr = &smV[src_offset];

    uint32_t r0, r1;
    ldmatrix_x2_trans(r0, r1, smem_ptr);

    // Write r0, r1 to gmem
    out[tid * 2 + 0] = r0;
    out[tid * 2 + 1] = r1;
}

int main()
{
    // Allocate output buffer: 32 threads × 2 uint32 = 64 uint32 = 256 bytes (matches B-fragment size).
    uint32_t *out_d;
    CK(cudaMalloc(&out_d, 64 * sizeof(uint32_t)));
    CK(cudaMemset(out_d, 0, 64 * sizeof(uint32_t)));

    ldmatrix_trans_probe_kernel<<<1, 32>>>(out_d);
    CK(cudaDeviceSynchronize());

    uint32_t out_h[64];
    CK(cudaMemcpy(out_h, out_d, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // ===== Compute expected: transpose of smV[0..7][0..31] FP8 =====
    // After transpose: result[n][k] where n in [0,8), k in [0,32).
    // For ldmatrix x2.trans b16: it transposes considering b16 elements.
    // So result is logically smV_transposed[n=0..7 in b16 = 0..15 in FP8 even+odd][k=0..7]
    // Actually let me lay it out per PTX:
    //   ldmatrix.trans takes 8x8 b16 matrix, transposes it → output is 8x8 b16.
    //   Per-thread output: each of 32 threads gets 2 b16 = 4 bytes = 1 uint32 per matrix.
    //   With x2: 2 matrices → 2 uint32 per thread.
    //
    // Expected layout per PTX (m8n8 trans):
    //   Result is stored with each thread holding values transposed from input.
    //   Thread groups: lanes 0..3, 4..7, ..., 28..31 hold rows 0..7 of the transposed matrix.
    //
    // Rather than analyze in advance, just DUMP what we got and compare to manual transpose.

    // ===== Compute reference: transpose smV[0..7][0..31] FP8 → expected[0..31][0..7] FP8 =====
    uint8_t smV_ref[BC * N];
    for (int k = 0; k < BC; k++)
        for (int n = 0; n < N; n++)
            smV_ref[k * N + n] = (uint8_t)((k * 13 + n * 7) & 0xFF);

    printf("=== smV[0..7][0..15] FP8 (first 8 rows × 16 bytes) ===\n");
    for (int k = 0; k < 8; k++) {
        printf("  k=%d: ", k);
        for (int n = 0; n < 16; n++) printf("%02x ", smV_ref[k * N + n]);
        printf("\n");
    }
    printf("\n");

    printf("=== ldmatrix output (32 threads × 2 uint32) ===\n");
    for (int t = 0; t < 32; t++) {
        printf("  t=%2d: %08x %08x\n", t, out_h[t * 2 + 0], out_h[t * 2 + 1]);
    }

    // ===== Try to match: expected after transpose =====
    // If ldmatrix.trans on first 8x16 (b16) gives 16x8 (b16) result.
    // 16 rows × 8 cols of b16 = 16 × 16 bytes = 256 bytes. Spread across 32 threads = 8 bytes/thread = 2 uint32 ✓.
    //
    // For matrix 0 (k=0..7, n=0..15 FP8 = n=0..7 b16):
    //   Input: smV[k][n_b16] where k in [0,8), n_b16 in [0,8)
    //   Transposed: result[n_b16][k] for n_b16 in [0,8), k in [0,8)
    //
    // For an 8x8 b16 matrix transposed via ldmatrix.trans, output layout:
    //   threads 0..7 → row 0 of result, columns 0..7 (each thread gets 1 b16)
    //   threads 8..15 → row 1
    //   ...
    //   threads 24..31 → row 3
    //   The remaining rows 4..7 of result are held in r1 (second register of x1) — but with x1 single matrix.
    // For x2 (2 matrices), r0 holds matrix 0, r1 holds matrix 1.
    // Actually wait — that's still wrong layout. Let me look at PTX spec.

    printf("\n=== Reference transposed layout — manual matrix 0 expected ===\n");
    // After ldmatrix.trans (single 8x8 b16 matrix), output is held by 32 threads as:
    //   Thread (i,j) for i in [0,4), j in [0,8): row i*2 + (j&1), col j>>1 → confusing.
    // Per Hopper PTX docs: for m8n8.trans, output matrix's element at row i col j is held by thread (j*4 + (i/2)), high or low half based on (i%2).
    // This is the same layout as the OUTPUT of mma.sync m16n8 D operand (row-major).
    //
    // Easier: just verify the BYTES match by enumerating what each thread should hold.

    // For thread t, holds 2 b16 values (matrix 0 only, r0). After transpose:
    //   t in 0..3: row 0 of transposed result, cols 0..3 (each thread one b16, low+high half)
    //   Actually for m8n8.x1.trans:
    //     Each thread holds 2 b16 elements: 1 from row (t/4)*2 col (t%4)*2 (low half) and row (t/4)*2 + 1 col (t%4)*2 (high half)
    // Confusing. Let me just print 16 expected results for thread 0 and 1 and check.

    // Bytes in r0 of thread 0 should be: smV_ref[?][?][0] and smV_ref[?][?][1] from transposed
    // Let me just dump the actual vs reference patterns for visual inspect.

    printf("\nIf transpose worked: thread 0's r0 should contain element from smV[*][0..1] in b16 packing.\n");
    printf("Specifically, row 0 col 0 of transposed = original [0][0..1] = smV[0][0] | (smV[0][1] << 8) = %02x | %02x = %04x\n",
           smV_ref[0], smV_ref[1], (smV_ref[1] << 8) | smV_ref[0]);
    printf("Got from kernel: thread 0 r0 = %08x  (low 16: %04x)\n",
           out_h[0], out_h[0] & 0xFFFF);

    cudaFree(out_d);
    return 0;
}
