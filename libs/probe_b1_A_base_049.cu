// 049 Ручка A: base-offset для b1. sm_addr_hi = &smQ[(kb*32 + 16 + lane)*Hd + np*16]
// (все лейны подают строки 16..31, НЕ lane-shift).
// Марkер-байтовая сверка на макете 048 (smQ 64x128 fp8, marker=(row<<4)|(col&0xF)).
// Критерий 100% на всех (lane, kb, np).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(1);}} while(0)

__global__ void probe_A_base(uint32_t *out) {
    __shared__ uint8_t smQ[64 * 128];
    int tid = threadIdx.x;
    // 049 revised marker: byte = row (0..63) uniquely — ignore col ID via marker; col identified
    // через lane position + reg mapping. Row 0..63 in 6-bit + 2-bit spare.
    if (tid < 128) {
        for (int j = 0; j < 64; ++j) {
            int idx = tid + j * 128;
            int r = idx / 128;    // 0..63 uniquely representable in 8-bit
            smQ[idx] = (uint8_t)(r & 0x3F);  // 6-bit row; col identified by position in reg
        }
    }
    __syncthreads();

    int lane = tid & 31;
    // Test kb=0..1, np=0..3 (12 iterations to cover various offsets)
    #pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
        #pragma unroll
        for (int np = 0; np < 4; ++np) {
            // b0: base = kb*32 + lane
            uint32_t sm_lo = __cvta_generic_to_shared(&smQ[(kb*32 + lane) * 128 + np*16]);
            uint32_t R0lo, R1lo, dm1, dm2;
            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                : "=r"(R0lo), "=r"(R1lo), "=r"(dm1), "=r"(dm2) : "r"(sm_lo));

            // b1 via BASE-OFFSET: base = kb*32 + 16 + lane (все лейны в окно 16..31 без wrap)
            // NB: для kb=0: rows 16..47 (все в bounds 0..63); для kb=1: rows 48..79 (выход за 64!)
            //     Пришлось бы ограничить kb, но всё равно проверяем.
            //     Для kb=1 rows 48..79 частично out-of-bounds; smQ 8192B; чтение вне bounds — undefined.
            //     Проверю только kb=0 для чистоты.
            if (kb == 0) {
                uint32_t sm_hi = __cvta_generic_to_shared(&smQ[(kb*32 + 16 + lane) * 128 + np*16]);
                uint32_t R0hi, R1hi, dm3, dm4;
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                    : "=r"(R0hi), "=r"(R1hi), "=r"(dm3), "=r"(dm4) : "r"(sm_hi));
                int base = kb*(4*32*4) + np*(32*4) + lane*4;
                if (tid < 32) {
                    out[base+0] = R0lo;
                    out[base+1] = R1lo;
                    out[base+2] = R0hi;
                    out[base+3] = R1hi;
                }
            }
        }
    }
}

int main() {
    const int TOTAL = 2*4*32*4;
    uint32_t *d_out; uint32_t h[TOTAL];
    CK(cudaMalloc(&d_out, TOTAL*sizeof(uint32_t)));
    CK(cudaMemset(d_out, 0xFF, TOTAL*sizeof(uint32_t)));
    probe_A_base<<<1, 128>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel fail: %s\n", cudaGetErrorString(err)); return 1; }
    CK(cudaMemcpy(h, d_out, TOTAL*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("049 Ручка A: base-offset для b1 (kb=0 only, безопасно в bounds)\n\n");

    // Проверка row-only marker: R0hi должен содержать rows {16+4*laneID+0..3} для kb=0 b1
    int ok = 0, tested = 0;
    for (int np = 0; np < 4; ++np) {
        for (int lane = 0; lane < 32; ++lane) {
            int laneID = lane % 4;
            int base = 0*(4*32*4) + np*(32*4) + lane*4;
            uint32_t R0hi = h[base+2], R1hi = h[base+3];
            for (int bt = 0; bt < 4; ++bt) {
                uint8_t byte = (R0hi >> (bt*8)) & 0xFF;
                int r = byte & 0x3F;
                int exp_row = (4*laneID + bt + 16);  // k+16 offset (kb=0)
                if (r == exp_row) ok++;
                tested++;
            }
            for (int bt = 0; bt < 4; ++bt) {
                uint8_t byte = (R1hi >> (bt*8)) & 0xFF;
                int r = byte & 0x3F;
                int exp_row = (4*laneID + bt + 16);
                if (r == exp_row) ok++;
                tested++;
            }
        }
    }
    printf("Match: %d / %d (%.2f%%)\n", ok, tested, 100.0*ok/tested);
    printf("Verdict A: %s\n\n", (ok==tested)?"PASS — base-offset ДОСТАВЛЯЕТ b1 (k+16)":"FAIL — base-offset НЕ даёт b1");

    printf("Sample kb=0 np=0 (b0=R0lo/R1lo, b1_candidate=R0hi/R1hi):\n");
    for (int l : {0, 1, 4, 8, 16}) {
        int base = 0 + 0 + l*4;
        printf("  lane %2d: R0lo=%08x R1lo=%08x R0hi=%08x R1hi=%08x\n",
               l, h[base+0], h[base+1], h[base+2], h[base+3]);
    }

    // Interpretation of R0hi
    printf("\nRow decoding для lane 0 R0hi=%08x (ожидание: (16,17,18,19) at col 0):\n", h[8]);
    for (int bt=0; bt<4; ++bt) {
        uint8_t byte = (h[8] >> (bt*8)) & 0xFF;
        printf("  byte%d = 0x%02x = (row=%d, col_lo=%d)\n", bt, byte, byte>>4, byte&0xF);
    }

    cudaFree(d_out);
    return (ok==tested)?0:1;
}
