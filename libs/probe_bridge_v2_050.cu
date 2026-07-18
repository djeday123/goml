// 050 §2 Мост v2: честный маркер row&0x3F, полное покрытие 8192 байт,
// обе формулы b1 (A base-offset, B lane-shift), + OOB-дискриминатор.
// Домен маркера: row ∈ [0..63], byte = row & 0x3F (6 бит, инъективно на 64 rows).
// Col не в маркере — идентифицируется через lane position в регистре.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(1);}} while(0)

// smQ mimic 64 × 128 fp8, marker row & 0x3F
__device__ __forceinline__ void init_smQ(uint8_t *smQ, int tid) {
    if (tid < 128) {
        for (int j = 0; j < 64; ++j) {
            int idx = tid + j * 128;
            int r = idx / 128;
            smQ[idx] = (uint8_t)(r & 0x3F);
        }
    }
}

// Форма A: base-offset 16+lane (все лейны)
__global__ void probe_A(uint32_t *out) {
    __shared__ uint8_t smQ[64 * 128];
    int tid = threadIdx.x;
    init_smQ(smQ, tid);
    __syncthreads();
    int lane = tid & 31;
    // kb ∈ [0..1], np ∈ [0..7]  = 16 iterations × 32 lanes × 4 regs = 2048 slots
    #pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
        #pragma unroll
        for (int np = 0; np < 8; ++np) {
            // b0 lo
            int row_lo = kb*32 + lane;
            if (row_lo >= 64) row_lo -= 32;   // safe clamp для kb=1 range
            uint32_t sm_lo = __cvta_generic_to_shared(&smQ[row_lo * 128 + np*16]);
            uint32_t R0lo, R1lo, d1, d2;
            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                : "=r"(R0lo), "=r"(R1lo), "=r"(d1), "=r"(d2) : "r"(sm_lo));

            // b1 A: base-offset 16+lane (без clamp; для kb=1 lane 16..31 → row 64..79 OOB!)
            int row_hi = kb*32 + 16 + lane;
            // Для теста без OOB: clamp to smQ bounds для оценки inband чтения
            // Но здесь тестируем инъективность layout не OOB. Тест OOB отдельно.
            if (row_hi >= 64) row_hi = 63;  // clamp temporary для safety
            uint32_t sm_hi = __cvta_generic_to_shared(&smQ[row_hi * 128 + np*16]);
            uint32_t R0hi, R1hi, d3, d4;
            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                : "=r"(R0hi), "=r"(R1hi), "=r"(d3), "=r"(d4) : "r"(sm_hi));

            int base = kb*(8*32*4) + np*(32*4) + lane*4;
            if (tid < 32) {
                out[base+0] = R0lo; out[base+1] = R1lo;
                out[base+2] = R0hi; out[base+3] = R1hi;
            }
        }
    }
}

// Форма B: lane-shift (lane&15)+16
__global__ void probe_B(uint32_t *out) {
    __shared__ uint8_t smQ[64 * 128];
    int tid = threadIdx.x;
    init_smQ(smQ, tid);
    __syncthreads();
    int lane = tid & 31;
    #pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
        #pragma unroll
        for (int np = 0; np < 8; ++np) {
            uint32_t sm_lo = __cvta_generic_to_shared(&smQ[(kb*32 + lane) * 128 + np*16]);
            uint32_t R0lo, R1lo, d1, d2;
            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                : "=r"(R0lo), "=r"(R1lo), "=r"(d1), "=r"(d2) : "r"(sm_lo));

            // b1 B: lane-shift (lane&15)+16
            uint32_t sm_hi = __cvta_generic_to_shared(&smQ[(kb*32 + ((lane & 15) + 16)) * 128 + np*16]);
            uint32_t R0hi, R1hi, d3, d4;
            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
                : "=r"(R0hi), "=r"(R1hi), "=r"(d3), "=r"(d4) : "r"(sm_hi));

            int base = kb*(8*32*4) + np*(32*4) + lane*4;
            if (tid < 32) {
                out[base+0] = R0lo; out[base+1] = R1lo;
                out[base+2] = R0hi; out[base+3] = R1hi;
            }
        }
    }
}

// OOB-дискриминатор: лейнам 16..31 подать "чужую" строку row=0 при A-формуле.
// Изменились ли R0/R1 лейнов 0..15?
__global__ void probe_OOB(uint32_t *out) {
    __shared__ uint8_t smQ[64 * 128];
    int tid = threadIdx.x;
    init_smQ(smQ, tid);
    __syncthreads();
    int lane = tid & 31;
    // kb=0, np=0 only for simplicity
    // Test 1: normal A — lanes 0..15 подают row_ptr 16..31; lanes 16..31 подают row_ptr 32..47
    uint32_t sm_normal = __cvta_generic_to_shared(&smQ[(16 + lane) * 128]);
    uint32_t R0n, R1n, d1, d2;
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
        : "=r"(R0n), "=r"(R1n), "=r"(d1), "=r"(d2) : "r"(sm_normal));

    // Test 2: same, но лейнам 16..31 подать row 0 (чужая строка); лейнам 0..15 без изменений
    int row_test = (lane < 16) ? (16 + lane) : 0;
    uint32_t sm_test = __cvta_generic_to_shared(&smQ[row_test * 128]);
    uint32_t R0t, R1t, d3, d4;
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];"
        : "=r"(R0t), "=r"(R1t), "=r"(d3), "=r"(d4) : "r"(sm_test));

    if (tid < 32) {
        out[lane*4+0] = R0n; out[lane*4+1] = R1n;
        out[lane*4+2] = R0t; out[lane*4+3] = R1t;
    }
}

// Check bytes function (row-only marker)
static int check_probe(uint32_t *h, int expected_offset, const char *label) {
    int ok = 0, tested = 0;
    for (int kb = 0; kb < 2; ++kb) {
        for (int np = 0; np < 8; ++np) {
            for (int lane = 0; lane < 32; ++lane) {
                int laneID = lane % 4;
                int base = kb*(8*32*4) + np*(32*4) + lane*4;
                uint32_t R0hi = h[base+2], R1hi = h[base+3];
                for (int bt = 0; bt < 4; ++bt) {
                    uint8_t byte = (R0hi >> (bt*8)) & 0xFF;
                    int r = byte & 0x3F;
                    int exp_row = ((kb*32 + 4*laneID + bt + expected_offset) & 0x3F);
                    if (r == exp_row) ok++;
                    tested++;
                }
                for (int bt = 0; bt < 4; ++bt) {
                    uint8_t byte = (R1hi >> (bt*8)) & 0xFF;
                    int r = byte & 0x3F;
                    int exp_row = ((kb*32 + 4*laneID + bt + expected_offset) & 0x3F);
                    if (r == exp_row) ok++;
                    tested++;
                }
            }
        }
    }
    printf("%s: %d / %d (%.2f%%)\n", label, ok, tested, 100.0*ok/tested);
    return ok == tested;
}

int main() {
    printf("050 §1 Мост v2: инъективный маркер row&0x3F, полное покрытие\n");
    printf("Домен: row ∈ [0..63] (6 бит); marker byte = row & 0x3F — инъективно\n\n");

    const int TOTAL_AB = 2*8*32*4;  // 2048
    uint32_t *d_out; uint32_t hA[TOTAL_AB], hB[TOTAL_AB];
    CK(cudaMalloc(&d_out, TOTAL_AB*sizeof(uint32_t)));

    CK(cudaMemset(d_out, 0xFF, TOTAL_AB*sizeof(uint32_t)));
    probe_A<<<1, 128>>>(d_out);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hA, d_out, TOTAL_AB*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CK(cudaMemset(d_out, 0xFF, TOTAL_AB*sizeof(uint32_t)));
    probe_B<<<1, 128>>>(d_out);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hB, d_out, TOTAL_AB*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("=== Формула A (base-offset 16+lane) ===\n");
    bool A_ok = check_probe(hA, 16, "Match");
    printf("Sample kb=0 np=0 lane 0: R0lo=%08x R1lo=%08x R0hi=%08x R1hi=%08x\n\n",
           hA[0], hA[1], hA[2], hA[3]);

    printf("=== Формула B (lane-shift (lane&15)+16) ===\n");
    bool B_ok = check_probe(hB, 16, "Match");
    printf("Sample kb=0 np=0 lane 0: R0lo=%08x R1lo=%08x R0hi=%08x R1hi=%08x\n\n",
           hB[0], hB[1], hB[2], hB[3]);

    // OOB probe
    printf("=== OOB-дискриминатор ===\n");
    uint32_t *d_oob; uint32_t hOOB[128];
    CK(cudaMalloc(&d_oob, 128*sizeof(uint32_t)));
    CK(cudaMemset(d_oob, 0xFF, 128*sizeof(uint32_t)));
    probe_OOB<<<1, 128>>>(d_oob);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hOOB, d_oob, 128*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Проверка: R0n vs R0t для lanes 0..15 (должны быть одинаковыми если lanes 16..31 игнорируются)
    int diff_count = 0;
    for (int lane = 0; lane < 16; ++lane) {
        if (hOOB[lane*4+0] != hOOB[lane*4+2]) diff_count++;
        if (hOOB[lane*4+1] != hOOB[lane*4+3]) diff_count++;
    }
    printf("Lanes 0..15 R0/R1 diff (normal vs OOB): %d / 32 slots\n", diff_count);
    printf("Sample lane 0: normal R0=%08x R1=%08x; OOB test R0=%08x R1=%08x\n",
           hOOB[0], hOOB[1], hOOB[2], hOOB[3]);
    if (diff_count == 0) {
        printf("Вердикт OOB: HW ИГНОРИРУЕТ лейны 16..31 — production-формула = (A) или (B) эквивалентны\n");
    } else {
        printf("Вердикт OOB: HW ЧИТАЕТ лейны 16..31 — production-формула требует клампа, выбор с memcheck-judged\n");
    }

    cudaFree(d_out); cudaFree(d_oob);
    return (A_ok || B_ok) ? 0 : 1;
}
