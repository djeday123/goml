// Diagnostic probe: ldmatrix.trans for FP8 PV MMA B-operand layout.
// Goal 1: verify ldmatrix.x4.trans.shared.b16 compiles + runs on sm_120a.
// Goal 2: dump per-thread loaded bytes, check pattern matches MMA expected B layout.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(e)); return 1; } } while (0)

__global__ void probe_kernel(uint8_t *out) {
    __shared__ __align__(16) uint8_t smV[512]; // 64 rows × 8 cols of bytes
    int tid = threadIdx.x;
    // Fill: byte at offset i = (uint8_t)i. Easy to verify what each thread loaded.
    if (tid < 16) {
        for (int i = 0; i < 32; i++) {
            int idx = tid * 32 + i;
            smV[idx] = (uint8_t)idx;
        }
    }
    __syncthreads();

    // ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
    // Loads 4 8x8 b16 matrices, applying transpose.
    // Per-thread addr: lane 0..7 → matrix 0 rows 0..7, lanes 8..15 → matrix 1, etc.
    int lane = tid & 31;
    int matrix_id = lane >> 3;       // 0..3
    int row_in_matrix = lane & 7;    // 0..7
    // Each 8x8 b16 matrix = 128 bytes. Each row = 8 b16 = 16 bytes.
    uint8_t *addr = &smV[matrix_id * 128 + row_in_matrix * 16];
    uint32_t sa = __cvta_generic_to_shared(addr);

    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sa));

    // Store 4 b32 per thread = 16 bytes
    uint32_t *out32 = (uint32_t *)out;
    out32[tid * 4 + 0] = r0;
    out32[tid * 4 + 1] = r1;
    out32[tid * 4 + 2] = r2;
    out32[tid * 4 + 3] = r3;
}

int main() {
    uint8_t *out_d;
    uint8_t out_h[32 * 16];
    CK(cudaMalloc(&out_d, 32 * 16));

    probe_kernel<<<1, 32>>>(out_d);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    CK(cudaMemcpy(out_h, out_d, 32 * 16, cudaMemcpyDeviceToHost));

    printf("=== ldmatrix.x4.trans probe on sm_120a ===\n");
    printf("SMEM fill: byte at offset i = (uint8_t)i\n");
    printf("Per-thread output (16 bytes = 4 b32 registers, after implicit trans):\n\n");
    for (int t = 0; t < 32; t++) {
        printf("tid %2d  (g%d t%d):  ", t, t / 4, t % 4);
        for (int i = 0; i < 16; i++) {
            printf("%02x ", out_h[t * 16 + i]);
            if (i % 4 == 3 && i < 15) printf("| ");
        }
        printf("\n");
    }
    cudaFree(out_d);
    return 0;
}
