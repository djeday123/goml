// 043 ISA-инвентарь ldmatrix на sm_120a.
//   Матрица: {m8n8.b16, m8n16.b8, m16n16.b8} × {no-trans, .trans} × {x1, x2, x4}
//   Для каждого live: маркер-байтовая карта (lane, reg, byte) <- (row, col)
//   Anti-DCE прием 013: результаты в глобаль.
//   Гвоздь 010 отделен: ISA-факт vs "компилятор не эмитит" через рукописный PTX.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(e)); }} while (0)

// ====== m8n8.b16 x4 no-trans (baseline, известно из 038) ======
__global__ void probe_m8n8_x4_notrans_b16(uint32_t *out) {
    __shared__ __half smem[16 * 16];
    int tid = threadIdx.x;
    if (tid < 128) {
        int idx0 = tid * 2, idx1 = idx0 + 1;
        int r0=idx0/16, c0=idx0%16, r1=idx1/16, c1=idx1%16;
        smem[idx0] = __short_as_half((short)((r0<<8)|c0));
        smem[idx1] = __short_as_half((short)((r1<<8)|c1));
    }
    __syncthreads();
    int lane = tid & 31;
    int row_ptr_lane = (lane < 16) ? lane : (lane - 16);
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row_ptr_lane * 16 + (lane >= 16 ? 8 : 0)]);
    uint32_t r0,r1,r2,r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(sm_addr));
    if (tid < 32) { out[tid*4+0]=r0; out[tid*4+1]=r1; out[tid*4+2]=r2; out[tid*4+3]=r3; }
}

// ====== m8n8.b16 x4 .trans (известно из 039) ======
__global__ void probe_m8n8_x4_trans_b16(uint32_t *out) {
    __shared__ __half smem[16 * 16];
    int tid = threadIdx.x;
    if (tid < 128) {
        int idx0 = tid * 2, idx1 = idx0 + 1;
        int r0=idx0/16, c0=idx0%16, r1=idx1/16, c1=idx1%16;
        smem[idx0] = __short_as_half((short)((r0<<8)|c0));
        smem[idx1] = __short_as_half((short)((r1<<8)|c1));
    }
    __syncthreads();
    int lane = tid & 31;
    int tile = lane / 8, rowid = lane % 8;
    int row = rowid + ((tile & 1) ? 8 : 0);
    int col_start = (tile & 2) ? 8 : 0;
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row * 16 + col_start]);
    uint32_t r0,r1,r2,r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(sm_addr));
    if (tid < 32) { out[tid*4+0]=r0; out[tid*4+1]=r1; out[tid*4+2]=r2; out[tid*4+3]=r3; }
}

// ====== m8n8.b16 x2 no-trans ======
__global__ void probe_m8n8_x2_notrans_b16(uint32_t *out) {
    __shared__ __half smem[8 * 16];
    int tid = threadIdx.x;
    if (tid < 64) {
        smem[tid*2+0] = __short_as_half((short)(((tid*2+0)/16 << 8) | ((tid*2+0)%16)));
        smem[tid*2+1] = __short_as_half((short)(((tid*2+1)/16 << 8) | ((tid*2+1)%16)));
    }
    __syncthreads();
    int lane = tid & 31;
    int row = lane % 8;
    int col_start = (lane / 8) * 8;
    if (col_start >= 16) col_start -= 16;
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row * 16 + col_start]);
    uint32_t r0,r1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
        : "=r"(r0),"=r"(r1) : "r"(sm_addr));
    if (tid < 32) { out[tid*2+0]=r0; out[tid*2+1]=r1; }
}

// ====== m8n8.b16 x2 .trans ======
__global__ void probe_m8n8_x2_trans_b16(uint32_t *out) {
    __shared__ __half smem[8 * 16];
    int tid = threadIdx.x;
    if (tid < 64) {
        smem[tid*2+0] = __short_as_half((short)(((tid*2+0)/16 << 8) | ((tid*2+0)%16)));
        smem[tid*2+1] = __short_as_half((short)(((tid*2+1)/16 << 8) | ((tid*2+1)%16)));
    }
    __syncthreads();
    int lane = tid & 31;
    int row = lane % 8;
    int col_start = (lane / 8) * 8;
    if (col_start >= 16) col_start -= 16;
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row * 16 + col_start]);
    uint32_t r0,r1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
        : "=r"(r0),"=r"(r1) : "r"(sm_addr));
    if (tid < 32) { out[tid*2+0]=r0; out[tid*2+1]=r1; }
}

// ====== m8n8.b16 x1 no-trans ======
__global__ void probe_m8n8_x1_notrans_b16(uint32_t *out) {
    __shared__ __half smem[8 * 8];
    int tid = threadIdx.x;
    if (tid < 32) {
        smem[tid*2+0] = __short_as_half((short)(((tid*2+0)/8 << 8) | ((tid*2+0)%8)));
        smem[tid*2+1] = __short_as_half((short)(((tid*2+1)/8 << 8) | ((tid*2+1)%8)));
    }
    __syncthreads();
    int lane = tid & 31;
    int row = lane % 8;
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row * 8]);
    uint32_t r0;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];"
        : "=r"(r0) : "r"(sm_addr));
    if (tid < 32) { out[tid]=r0; }
}

// ====== m8n8.b16 x1 .trans ======
__global__ void probe_m8n8_x1_trans_b16(uint32_t *out) {
    __shared__ __half smem[8 * 8];
    int tid = threadIdx.x;
    if (tid < 32) {
        smem[tid*2+0] = __short_as_half((short)(((tid*2+0)/8 << 8) | ((tid*2+0)%8)));
        smem[tid*2+1] = __short_as_half((short)(((tid*2+1)/8 << 8) | ((tid*2+1)%8)));
    }
    __syncthreads();
    int lane = tid & 31;
    int row = lane % 8;
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row * 8]);
    uint32_t r0;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0},[%1];"
        : "=r"(r0) : "r"(sm_addr));
    if (tid < 32) { out[tid]=r0; }
}

// ====== m8n16.b8x16.b6x16_p32 (Blackwell packed FP8 shape) — попытка синтаксиса ======
// (skipped — requires packed subbyte types; отдельная проба)

// ====== m16n16.b8 x2 .trans (per ptxas: требует 4 output uint32, не 2) ======
__global__ void probe_m16n16_x2_trans_b8(uint32_t *out) {
    __shared__ uint8_t smem[16 * 32];  // 16 rows × 32 cols = 512 bytes
    int tid = threadIdx.x;
    if (tid < 512) { smem[tid] = (uint8_t)((tid % 32) | ((tid / 32) << 4)); }
    __syncthreads();
    int lane = tid & 31;
    int row = lane % 16;
    int col_base = (lane / 16) * 16;
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[row * 32 + col_base]);
    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sm_addr));
    if (tid < 32) { out[tid*4+0]=r0; out[tid*4+1]=r1; out[tid*4+2]=r2; out[tid*4+3]=r3; }
}

// Helper для дампа результатов
static void dump_result(const char *label, uint32_t *h_out, int per_lane) {
    printf("--- %s ---\n", label);
    for (int lane = 0; lane < 32; ++lane) {
        printf("l%02d:", lane);
        for (int r = 0; r < per_lane; ++r) {
            uint32_t v = h_out[lane * per_lane + r];
            uint16_t lo = v & 0xFFFF, hi = (v >> 16) & 0xFFFF;
            int rl = (lo >> 8) & 0xFF, cl = lo & 0xFF;
            int rh = (hi >> 8) & 0xFF, ch = hi & 0xFF;
            printf(" R%d=[(%d,%d)|(%d,%d)]", r, rl, cl, rh, ch);
        }
        printf("\n");
    }
}

// Dump raw bytes (for FP8 probes)
static void dump_raw(const char *label, uint32_t *h_out, int per_lane) {
    printf("--- %s (raw hex bytes per uint32) ---\n", label);
    for (int lane = 0; lane < 32; ++lane) {
        printf("l%02d:", lane);
        for (int r = 0; r < per_lane; ++r) {
            uint32_t v = h_out[lane * per_lane + r];
            printf(" R%d=%02x_%02x_%02x_%02x", r, v>>24, (v>>16)&0xFF, (v>>8)&0xFF, v&0xFF);
        }
        printf("\n");
    }
}

int main() {
    uint32_t *d_out;
    uint32_t h_out[128];
    CK(cudaMalloc(&d_out, 128 * sizeof(uint32_t)));

    #define TRY_KERNEL(name, kern, per_lane, is_fp8) do { \
        CK(cudaMemset(d_out, 0xFF, 128 * sizeof(uint32_t))); \
        (kern)<<<1, 128>>>(d_out); \
        cudaError_t err = cudaDeviceSynchronize(); \
        printf("\n===== %s =====\n", name); \
        if (err != cudaSuccess) { \
            printf("VERDICT: RUNTIME_ERROR: %s\n", cudaGetErrorString(err)); \
        } else { \
            printf("VERDICT: COMPILED_AND_RAN\n"); \
            CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost)); \
            if (is_fp8) dump_raw(name, h_out, per_lane); \
            else dump_result(name, h_out, per_lane); \
        } \
    } while(0)

    TRY_KERNEL("m8n8.x4.notrans.b16", probe_m8n8_x4_notrans_b16, 4, 0);
    TRY_KERNEL("m8n8.x4.trans.b16",   probe_m8n8_x4_trans_b16,   4, 0);
    TRY_KERNEL("m8n8.x2.notrans.b16", probe_m8n8_x2_notrans_b16, 2, 0);
    TRY_KERNEL("m8n8.x2.trans.b16",   probe_m8n8_x2_trans_b16,   2, 0);
    TRY_KERNEL("m8n8.x1.notrans.b16", probe_m8n8_x1_notrans_b16, 1, 0);
    TRY_KERNEL("m8n8.x1.trans.b16",   probe_m8n8_x1_trans_b16,   1, 0);
    TRY_KERNEL("m16n16.x2.trans.b8",  probe_m16n16_x2_trans_b8, 4, 1);

    cudaFree(d_out);
    return 0;
}
