// 071 x1-probe: standalone проба ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8
// на макете smQ-раскладки production (свизл swz_byte дословно из fa_bwd_common.cuh).
//
// Цели:
//   (a) Компилируется/исполняется на sm_120a?
//   (b) Маркер-байтовая инъективность: доставка == b0-pair как ожидает MMA-B?
//   (c) Сколько выходных регистров? Есть ли дубликаты?
//   (d) Полное покрытие MMA-B требует 32 x2 → 64 x1 = +32 выдачи/qt в очередь dk_new.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "fa_bwd_common.cuh"

#define CK(c) do{auto e=(c); if(e!=cudaSuccess){printf("CUDA err %d @%d\n",(int)e,__LINE__); return 1;}}while(0)

// Prod dk_new: Br=64, Bc=64, Hd=128, THREADS=128, warps=4 warps of 32 lanes.
// smQ layout: 8192 B, свизл swz_byte(i_local, col_byte).
// LDSM.trans.b8 читает b8 транспонированные фрагменты для MMA-B m16n8k32.e4m3.

__global__ void probe_x1(uint8_t *smQ_shadow, uint32_t *out_regs, uint32_t *out_addrs)
{
    __shared__ uint8_t smQ[8192];   // 64 rows × 128 cols = same as prod dk_new smQ

    int tid = threadIdx.x;
    int wid = tid >> 5;      // warp id 0..3
    int lane = tid & 31;
    int l_div4 = lane >> 2;
    int l_mod4 = lane & 3;

    // Writer: swizzled smQ с маркером byte@(row, col) = row (injective by row).
    // Row-only marker избегает col-aliasing (уроки 058 col-map).
    constexpr int Br = 64;
    constexpr int Hd = 128;
    constexpr int CHUNK = 16;
    constexpr int cpr = Hd / CHUNK;
    constexpr int total = Br * cpr;
    for (int c = tid; c < total; c += 128) {
        int row = c / cpr;
        int col_byte = (c % cpr) * CHUNK;
        int dst = swz_byte(row, col_byte);
        // маркер byte@(row,col) = row (0..63)
        uint8_t marker = (uint8_t)row;
        for (int b = 0; b < CHUNK; ++b) {
            smQ[dst + b] = marker;
        }
    }
    __syncthreads();

    // Reader: LDSM.x1.trans.b8. Пробуем формулу row_ptr production (kb=0 case).
    // Мост 060-B / 061 для x2:
    //   row_lo = kb*32 + lane           (lo — b0 slot)
    //   row_hi = kb*32 + (lane&15) + 16 (hi — b1 slot)
    // Для x1 предположим ту же формулу row_lo (одна матрица за раз).
    int kb = 0;
    int np = 0;   // ni-pair 0 → выходы для ni_a=0, ni_b=1
    int row_lo = kb * 32 + lane;
    int addr_lo = swz_byte(row_lo, np * 16);
    uint32_t sm_addr_lo = __cvta_generic_to_shared(&smQ[addr_lo]);

    // Пытаемся LDSM.x1.trans.b8 с 2 output-регистрами (типовая раскладка для m16n16 x1).
    uint32_t R0 = 0xDEADBEEFu, R1 = 0xDEADBEEFu;
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1},[%2];\n"
        : "=r"(R0), "=r"(R1)
        : "r"(sm_addr_lo));

    // Дампим 2 регистра per lane + адрес.
    int slot = tid;
    out_regs[slot * 2 + 0] = R0;
    out_regs[slot * 2 + 1] = R1;
    out_addrs[slot] = addr_lo;
}

int main() {
    printf("=== 071 x1-probe ===\n");
    printf("target: ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%%0,%%1}\n");
    printf("layout: swz_byte(row, col_byte) writer + row-marker (byte=row)\n\n");

    uint8_t *d_smQ_shadow;
    uint32_t *d_regs, *d_addrs;
    CK(cudaMalloc(&d_smQ_shadow, 8192));
    CK(cudaMalloc(&d_regs, 128 * 2 * 4));
    CK(cudaMalloc(&d_addrs, 128 * 4));

    probe_x1<<<1, 128>>>(d_smQ_shadow, d_regs, d_addrs);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("KERNEL FAILED: %s\n", cudaGetErrorString(e));
        return 1;
    }

    uint32_t regs[256], addrs[128];
    CK(cudaMemcpy(regs, d_regs, sizeof(regs), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(addrs, d_addrs, sizeof(addrs), cudaMemcpyDeviceToHost));

    // Дамп первого warp (wid=0), первый np, lane 0..31
    printf("wid=0 np=0 kb=0: lane -> R0 R1 addr (row-marker inside byte fragments)\n");
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t R0 = regs[lane * 2 + 0];
        uint32_t R1 = regs[lane * 2 + 1];
        uint32_t addr = addrs[lane];
        printf("  lane%2d: R0=%08x R1=%08x addr=%u\n", lane, R0, R1, addr);
    }

    // Инъективность: для x1 m16n16.trans.b8 ожидаем 2 uint32 per lane.
    // Каждый uint32 = 4 bytes = 4 marker'ов из свизлованных row'ов.
    // Проверяем: сумма unique выходных row-маркеров == 16 (m16n16 = 16 rows × 16 cols).
    printf("\n=== Injectivity: unique row-markers in wid=0 warp ===\n");
    bool seen[64] = {0};
    int uniq = 0;
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t R0 = regs[lane * 2 + 0];
        uint32_t R1 = regs[lane * 2 + 1];
        for (int shft = 0; shft < 32; shft += 8) {
            uint8_t b = (R0 >> shft) & 0xff;
            if (!seen[b]) { seen[b] = true; uniq++; }
            b = (R1 >> shft) & 0xff;
            if (!seen[b]) { seen[b] = true; uniq++; }
        }
    }
    printf("unique row-marker bytes seen in warp0 R0+R1: %d (expected 16 for m16n16.x1 one matrix)\n", uniq);
    printf("verdict: %s\n", (uniq == 16) ? "MATCHES m16n16 ONE MATRIX" : ((uniq == 32) ? "TWO MATRICES (dup or x2-like)" : "UNKNOWN"));

    // Также подсчитаем сколько duplicate pairs (R0 == R1 per lane).
    int dupR0R1 = 0;
    for (int lane = 0; lane < 32; ++lane) {
        if (regs[lane * 2 + 0] == regs[lane * 2 + 1]) dupR0R1++;
    }
    printf("lanes with R0==R1 in warp0: %d/32\n", dupR0R1);

    cudaFree(d_smQ_shadow); cudaFree(d_regs); cudaFree(d_addrs);
    printf("done\n");
    return 0;
}
