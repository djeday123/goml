// E1 Probe C: scheduler-цена ldmatrix.x2.trans vs 2×LDS
// PV-паттерн: 16 nt × (загрузить B + 2×QMMA для M_TILES=2)
// Замер cycles через clock64. Порог: ldmatrix-вариант должен быть НЕ медленнее +5%.
//
// V layout в smem:
//   v121 (LDS): smV_T [hd_outer][kv_inner] swizzled — B col-major reads напрямую
//   v130 (ldmatrix.trans): smV [kv_outer][hd_inner] natural — ldmatrix.x2.trans
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s\n", cudaGetErrorString(e)); return 1;}}while(0)

#define FA_STRIDE 128
#define SMV_T_STRIDE 68
#define M_TILES 2
#define N_LOOP_ITERS 5000  // достаточно для усреднения

__device__ __forceinline__ int swz_byte(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}
__device__ __forceinline__ int swz_byte_smvt(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * SMV_T_STRIDE + ((chunk ^ (row & 3)) << 4) + within;
}

__device__ __forceinline__ void mma_fp8_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t c0, uint32_t c1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

// === Variant 1: 2×LDS из smV_T (v121-style) ===
__global__ void probe_lds(uint64_t *cyc_out, uint8_t *gmem_init)
{
    __shared__ uint8_t smV_T[128 * SMV_T_STRIDE + 64];  // 8.5K + slack
    int tid = threadIdx.x;
    int lane = tid % 32;
    int gid = lane / 4;
    int t = lane % 4;
    // Init smV_T
#pragma unroll
    for (int i = tid; i < 128 * SMV_T_STRIDE; i += 128)
        smV_T[i] = gmem_init[i % 256];
    __syncthreads();

    uint32_t a0=0x12345678, a1=0xDEADBEEF, a2=0xCAFEBABE, a3=0x55AA33CC;
    uint32_t d_mi[M_TILES][2] = {{0,0},{0,0}};
    uint64_t t0 = clock64();
#pragma unroll 1
    for (int loop = 0; loop < N_LOOP_ITERS; loop++) {
#pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int br = nt * 8;
            int cl = t * 4;
            int ch = cl + 16;
            uint32_t b0 = *(uint32_t*)&smV_T[swz_byte_smvt(br + gid, cl)];
            uint32_t b1 = *(uint32_t*)&smV_T[swz_byte_smvt(br + gid, ch)];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++) {
                mma_fp8_f16(d_mi[mi][0], d_mi[mi][1],
                            a0, a1, a2, a3, b0, b1,
                            d_mi[mi][0], d_mi[mi][1]);
            }
            // Минимальное обновление a, чтобы компилятор не выкинул MMA
            a0 ^= d_mi[0][0];
            a1 ^= d_mi[0][1];
        }
    }
    uint64_t t1 = clock64();
    // Store sink
    if (tid == 0) {
        cyc_out[0] = t1 - t0;
        cyc_out[1] = (uint64_t)(d_mi[0][0] ^ d_mi[1][1]);  // sink
    }
}

// === Variant 2: ldmatrix.x2.trans из smV (v130-style natural) ===
__global__ void probe_ldmat(uint64_t *cyc_out, uint8_t *gmem_init)
{
    __shared__ uint8_t smV[64 * FA_STRIDE + 64];  // 8K + slack
    int tid = threadIdx.x;
    int lane = tid % 32;
    int gid = lane / 4;
    int t = lane % 4;
    // Init smV
#pragma unroll
    for (int i = tid; i < 64 * FA_STRIDE; i += 128)
        smV[i] = gmem_init[i % 256];
    __syncthreads();

    uint32_t a0=0x12345678, a1=0xDEADBEEF, a2=0xCAFEBABE, a3=0x55AA33CC;
    uint32_t d_mi[M_TILES][2] = {{0,0},{0,0}};
    uint64_t t0 = clock64();
#pragma unroll 1
    for (int loop = 0; loop < N_LOOP_ITERS; loop++) {
#pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            // ldmatrix.x2.trans даёт 2 b32/lane = ровно B m16k32 фрагмент
            // Адресация: 8 lanes (gid 0..7) подают адрес матрицы 8×8 b16
            // Матрица 1: rows kv=br+t*4..+3, cols hd_g (8 cols of hd)
            // smV[kv][hd] natural: addr = (br + lane_row) * 128 + hd_start + lane_col*2 (b16)
            // Для x2.trans с swizzle smV: row=br+gid (8 rows), col_bytes=t*16 (16 b16 = 32 bytes)
            // ldmatrix.x2.trans: 2 matrix 8×8 b16. 16 lanes подают addr (lanes 0..7 + 8..15).
            // Каждый адрес = начало строки 8×16 bytes b16 = 8 row × 8 b16 = 8 rows × 16 bytes.
            // Для probe: эмулируем nt-iter — берём kv-row из 64-row smV с wrap.
            int kv_row = (nt * 2 + (lane & 7)) & 63;
            int hd_col_b = (nt * 16) & 127;  // 16-byte aligned, wrap-around
            uint32_t sm_addr = __cvta_generic_to_shared(
                &smV[kv_row * FA_STRIDE + hd_col_b]);
            uint32_t b0, b1;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 {%0, %1}, [%2];\n"
                : "=r"(b0), "=r"(b1) : "r"(sm_addr));
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++) {
                mma_fp8_f16(d_mi[mi][0], d_mi[mi][1],
                            a0, a1, a2, a3, b0, b1,
                            d_mi[mi][0], d_mi[mi][1]);
            }
            a0 ^= d_mi[0][0];
            a1 ^= d_mi[0][1];
        }
    }
    uint64_t t1 = clock64();
    if (tid == 0) {
        cyc_out[0] = t1 - t0;
        cyc_out[1] = (uint64_t)(d_mi[0][0] ^ d_mi[1][1]);
    }
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s (CC %d.%d)\n", p.name, p.major, p.minor);

    uint64_t *cyc_d; uint8_t *init_d;
    CK(cudaMalloc(&cyc_d, 2 * sizeof(uint64_t)));
    CK(cudaMalloc(&init_d, 256));
    CK(cudaMemset(init_d, 0x42, 256));

    uint64_t lds_cyc, ldmat_cyc, sink;
    // Warmup
    probe_lds<<<1, 128>>>(cyc_d, init_d);
    probe_ldmat<<<1, 128>>>(cyc_d, init_d);
    CK(cudaDeviceSynchronize());

    // Measure 3 runs each, median
    uint64_t lds_runs[5], ldmat_runs[5];
    for (int r = 0; r < 5; r++) {
        probe_lds<<<1, 128>>>(cyc_d, init_d);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&lds_cyc, cyc_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&sink, cyc_d+1, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        lds_runs[r] = lds_cyc;

        probe_ldmat<<<1, 128>>>(cyc_d, init_d);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&ldmat_cyc, cyc_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&sink, cyc_d+1, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        ldmat_runs[r] = ldmat_cyc;
    }

    auto med5 = [](uint64_t *a){
        // sort 5 elements
        for (int i = 0; i < 5; i++)
            for (int j = i+1; j < 5; j++)
                if (a[j] < a[i]) { uint64_t tmp=a[i]; a[i]=a[j]; a[j]=tmp; }
        return a[2];
    };
    uint64_t lds_med = med5(lds_runs);
    uint64_t ldmat_med = med5(ldmat_runs);

    printf("\n=== Cycles (median of 5 runs, N_LOOP_ITERS=%d × 16 nt × 2 mi) ===\n",
           N_LOOP_ITERS);
    printf("  LDS:       %llu cycles (per nt-iter: %llu)\n",
           (unsigned long long)lds_med,
           (unsigned long long)(lds_med / (N_LOOP_ITERS * 16)));
    printf("  ldmatrix:  %llu cycles (per nt-iter: %llu)\n",
           (unsigned long long)ldmat_med,
           (unsigned long long)(ldmat_med / (N_LOOP_ITERS * 16)));
    double delta_pct = 100.0 * ((double)ldmat_med - (double)lds_med) / lds_med;
    printf("  Δ ldmatrix vs LDS: %+.2f%%\n", delta_pct);
    printf("\n");
    if (delta_pct > 5.0) {
        printf("=== VERDICT: RED — ldmatrix медленнее >5%%. v130 закрывается на E1. ===\n");
        return 1;
    } else if (delta_pct < -1.0) {
        printf("=== VERDICT: GREEN — ldmatrix быстрее. E2 зелёный. ===\n");
        return 0;
    } else {
        printf("=== VERDICT: GREEN (паритет ±5%%) — E2 зелёный. ===\n");
        return 0;
    }
}
