// =====================================================================
//  probe_m16n8k16_fp16_layout.cu — B2.1 prep probe.
//
//  ЦЕЛЬ: измерить (не верить докам) fragment layout
//        mma.sync.m16n8k16.row.col.f32.f16.f16.f32 на sm_120a
//        для выбора B-операнд адресации в backward dV.
//
//  Контекст: на этой карте уже были сюрпризы (a1↔a2 swap, setmaxnreg no-op,
//  ldmatrix.trans −55%). Перед телом B2.1 — изолированный тест.
//
//  ОПЕРАЦИЯ: D = Pᵀ · dO  (та же что в backward dV)
//      P size [16, 16] FP16 → P^T = A operand m16n8k16
//      dO size [16, 8] FP16 → B operand m16n8k16
//      D size [16, 8] FP32
//
//  ДВЕ ТЕСТИРУЕМЫЕ РАСКЛАДКИ B-ОПЕРАНДА:
//      B1: smdO row-major [k=16, n=8] (как лежит после upload, no transpose)
//          + manual 2× LDS.U16 + pack для каждого uint32 b_i (k-pair stride 16 B)
//      B2: smdO_T row-major [n=8, k=16] (dO транспонирован при upload)
//          + manual 1× LDS.U32 для каждого uint32 b_i (k-pair adjacent)
//
//  A-ОПЕРАНД ОДИНАКОВ для обоих путей: smP_T row-major [m=16, k=16]
//      (P записан транспонированно по директиве Vugar Q1).
//      Lane l читает 4 uint32 a_i через LDS.U32 (a-операнд row-major natively).
//
//  ПРИЁМ: per-lane выходные d0..d3 сравниваются с CPU-вычисленным D[m, n].
//      Если B1 и B2 оба сходятся → корректность не зависит от пути, выбор
//        по перф (B1: 2× LDS-bw в hot-loop; B2: full smdO transpose pass).
//      Если только B2 сходится → docs соврали на B1-pattern, путь α forced.
//      Если только B1 → сюрприз вкуса a1↔a2, переписываем модель.
//      Если оба врут → серьёзная проблема fragment layout, разбор отдельно.
//
//  АРТЕФАКТ: этот файл остаётся в libs/, переиспользуется для dK/dQ MMA
//  вопросов (там m16n8k32 fp8 и/или m16n8k16 fp16 в других ролях).
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e)); exit(1); }} while (0)

// PTX m16n8k16.row.col.f32.f16.f16.f32 — 1 warp выполняет одну MMA.
// A: 4 uint32 = 8 fp16  (m=16, k=16)
// B: 2 uint32 = 4 fp16  (k=16, n=8)
// D: 4 fp32             (m=16, n=8)
__device__ __forceinline__ void mma_m16n8k16_f16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// =====================================================================
// Probe kernel — runs BOTH paths back-to-back, writes D outputs per path.
//
// Smem layout:
//   smP_T  [16][16] half  = 512 B   — P stored transposed (smP_T[j,i]=P[i,j])
//   smdO_B1[16][ 8] half  = 256 B   — dO row-major (path B1)
//   smdO_B2[ 8][16] half  = 256 B   — dO transposed (path B2)
//   D_B1[4*32] = D_B2[4*32] FP32 в gmem (через kernel parameters)
// =====================================================================
__global__ void probe_kernel(
    const __half * __restrict__ smP_T_init,    // [16,16] = 256 half (gmem source)
    const __half * __restrict__ smdO_init,     // [16, 8] = 128 half (gmem source row-major)
    const __half * __restrict__ smdO_T_init,   // [ 8,16] = 128 half (gmem source transposed)
    float * __restrict__ D_B1_out,             // [32 lanes × 4 fp32]
    float * __restrict__ D_B2_out)
{
    __shared__ __half smP_T [16 * 16];
    __shared__ __half smdO  [16 *  8];
    __shared__ __half smdO_T[ 8 * 16];

    const int tid = threadIdx.x;
    const int lane = tid & 31;       // 0..31 (one warp probe)
    const int l_div4 = lane >> 2;    // 0..7
    const int l_mod4 = lane & 3;     // 0..3

    // cooperative SMEM load
    if (tid < 32) {
        #pragma unroll
        for (int e = 0; e < 8; ++e)
            smP_T[tid * 8 + e] = smP_T_init[tid * 8 + e];   // 32*8=256 = full
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            smdO[tid * 4 + e]  = smdO_init [tid * 4 + e];   // 32*4=128 = full
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            smdO_T[tid * 4 + e] = smdO_T_init[tid * 4 + e]; // same size
    }
    __syncthreads();

    // ----- A-operand load (same for both paths): P^T row-major from smP_T -----
    // Lane l holds (m, k) tiles:
    //   a0: m=(l/4)+0, k=(l%4)*2 + 0..1
    //   a1: m=(l/4)+8, k=(l%4)*2 + 0..1
    //   a2: m=(l/4)+0, k=(l%4)*2 + 8..9
    //   a3: m=(l/4)+8, k=(l%4)*2 + 8..9
    //
    // smP_T[m, k] row-major: addr = (m*16 + k)*2 bytes.
    // Pair (k, k+1) is adjacent → single LDS.U32 packs (k, k+1) half2.
    uint32_t a0, a1, a2, a3;
    {
        int m_lo = l_div4 + 0;
        int m_hi = l_div4 + 8;
        int k_lo = l_mod4 * 2 + 0;
        int k_hi = l_mod4 * 2 + 8;
        a0 = *reinterpret_cast<uint32_t*>(&smP_T[m_lo * 16 + k_lo]);
        a1 = *reinterpret_cast<uint32_t*>(&smP_T[m_hi * 16 + k_lo]);
        a2 = *reinterpret_cast<uint32_t*>(&smP_T[m_lo * 16 + k_hi]);
        a3 = *reinterpret_cast<uint32_t*>(&smP_T[m_hi * 16 + k_hi]);
    }

    // ====================================================================
    // PATH B1: dO row-major in smdO, 2× LDS.U16 + pack per uint32 b_i.
    //
    // Lane l holds:
    //   b0: k=(l%4)*2 + 0..1, n=l/4
    //   b1: k=(l%4)*2 + 8..9, n=l/4
    //
    // In smdO[k, n] row-major [16, 8]: addr = (k*8 + n)*2 bytes.
    // Pair (k=k0, k=k0+1) at same n → strides 16 B → NOT adjacent →
    //   2× LDS.U16 + manual pack.
    // ====================================================================
    uint32_t b0_B1, b1_B1;
    {
        int n   = l_div4;
        int k0a = l_mod4 * 2 + 0;
        int k0b = l_mod4 * 2 + 1;
        int k1a = l_mod4 * 2 + 8;
        int k1b = l_mod4 * 2 + 9;
        uint16_t lo0 = *reinterpret_cast<uint16_t*>(&smdO[k0a * 8 + n]);
        uint16_t hi0 = *reinterpret_cast<uint16_t*>(&smdO[k0b * 8 + n]);
        uint16_t lo1 = *reinterpret_cast<uint16_t*>(&smdO[k1a * 8 + n]);
        uint16_t hi1 = *reinterpret_cast<uint16_t*>(&smdO[k1b * 8 + n]);
        b0_B1 = ((uint32_t)hi0 << 16) | (uint32_t)lo0;
        b1_B1 = ((uint32_t)hi1 << 16) | (uint32_t)lo1;
    }

    // ====================================================================
    // PATH B2: dO transposed in smdO_T, single LDS.U32 per b_i.
    //
    // In smdO_T[n, k] row-major [8, 16]: addr = (n*16 + k)*2 bytes.
    // Pair (k=k0, k=k0+1) at same n adjacent in memory (stride 2 B).
    // ====================================================================
    uint32_t b0_B2, b1_B2;
    {
        int n  = l_div4;
        int k0 = l_mod4 * 2 + 0;
        int k1 = l_mod4 * 2 + 8;
        b0_B2 = *reinterpret_cast<uint32_t*>(&smdO_T[n * 16 + k0]);
        b1_B2 = *reinterpret_cast<uint32_t*>(&smdO_T[n * 16 + k1]);
    }

    // ----- Run MMA for both paths (D += A·B, C=0) -----
    float d0_B1=0.f, d1_B1=0.f, d2_B1=0.f, d3_B1=0.f;
    float d0_B2=0.f, d1_B2=0.f, d2_B2=0.f, d3_B2=0.f;
    mma_m16n8k16_f16_f32(d0_B1, d1_B1, d2_B1, d3_B1,
                         a0, a1, a2, a3,
                         b0_B1, b1_B1,
                         0.f, 0.f, 0.f, 0.f);
    mma_m16n8k16_f16_f32(d0_B2, d1_B2, d2_B2, d3_B2,
                         a0, a1, a2, a3,
                         b0_B2, b1_B2,
                         0.f, 0.f, 0.f, 0.f);

    // ----- Store outputs (lane × 4 fp32) -----
    if (tid < 32) {
        D_B1_out[lane * 4 + 0] = d0_B1;
        D_B1_out[lane * 4 + 1] = d1_B1;
        D_B1_out[lane * 4 + 2] = d2_B1;
        D_B1_out[lane * 4 + 3] = d3_B1;
        D_B2_out[lane * 4 + 0] = d0_B2;
        D_B2_out[lane * 4 + 1] = d1_B2;
        D_B2_out[lane * 4 + 2] = d2_B2;
        D_B2_out[lane * 4 + 3] = d3_B2;
    }
}

// =====================================================================
// CPU reference: D = P^T · dO. P[16,16], dO[16,8], D[16,8].
// =====================================================================
static void cpu_pt_dO(const float *P, const float *dO, float *D)
{
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k)
                acc += P[k * 16 + m] * dO[k * 8 + n];   // P[k,m] = P^T[m,k]
            D[m * 8 + n] = acc;
        }
    }
}

// Convert lane (m, n) ownership per documented mapping:
//   d0: m=(l/4)+0, n=(l%4)*2 + 0
//   d1: m=(l/4)+0, n=(l%4)*2 + 1
//   d2: m=(l/4)+8, n=(l%4)*2 + 0
//   d3: m=(l/4)+8, n=(l%4)*2 + 1
static int verify_path(const char *name, const float *D_gpu, const float *D_cpu)
{
    int mismatches = 0;
    double max_abs = 0.0;
    int worst_lane = -1, worst_slot = -1;
    float worst_gpu = 0.f, worst_cpu = 0.f;

    for (int lane = 0; lane < 32; ++lane) {
        int l_div4 = lane >> 2;
        int l_mod4 = lane & 3;
        int m_lo = l_div4 + 0;
        int m_hi = l_div4 + 8;
        int n_lo = l_mod4 * 2 + 0;
        int n_hi = l_mod4 * 2 + 1;

        struct { int m, n, slot; } map[4] = {
            { m_lo, n_lo, 0 },
            { m_lo, n_hi, 1 },
            { m_hi, n_lo, 2 },
            { m_hi, n_hi, 3 },
        };
        for (int e = 0; e < 4; ++e) {
            float gpu = D_gpu[lane * 4 + map[e].slot];
            float cpu = D_cpu[map[e].m * 8 + map[e].n];
            double abs_err = std::fabs((double)gpu - (double)cpu);
            double rel = abs_err / (std::fabs((double)cpu) + 1e-30);
            // Tight: FP16 mul + FP32 acc — abs ~1e-2 from FP16 floor on 16-elem sum
            if (abs_err > 5e-2 + 1e-2 * std::fabs((double)cpu)) {
                mismatches++;
            }
            if (abs_err > max_abs) {
                max_abs = abs_err;
                worst_lane = lane;
                worst_slot = map[e].slot;
                worst_gpu = gpu;
                worst_cpu = cpu;
            }
            (void)rel;
        }
    }
    printf("  %-12s mismatches=%-3d  max_abs=%.4e  worst@lane=%d,slot=%d  gpu=%.4f  cpu=%.4f\n",
           name, mismatches, max_abs, worst_lane, worst_slot, worst_gpu, worst_cpu);
    return mismatches;
}

int main()
{
    // Use FP32 host arrays, then cast to FP16 for upload.
    float P [16 * 16], dO  [16 *  8];
    float Pt[16 * 16], dOt [ 8 * 16];

    // Fill with values that fit nicely in FP16 (no overflow, exact roundtrip):
    //   P[i, j]   = (i - 7.5) * 0.25 + j * 0.0625     range ~[-1.875, 1.875]
    //   dO[i, d]  = (d - 3.5) * 0.5  + i * 0.03125    range ~[-1.75, 2.25]
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            P[i * 16 + j] = ((float)i - 7.5f) * 0.25f + (float)j * 0.0625f;
    for (int i = 0; i < 16; ++i)
        for (int d = 0; d < 8; ++d)
            dO[i * 8 + d] = ((float)d - 3.5f) * 0.5f + (float)i * 0.03125f;

    // smP_T layout: smP_T[m=j, k=i] = P[i, j]  (transposed)
    for (int m = 0; m < 16; ++m)
        for (int k = 0; k < 16; ++k)
            Pt[m * 16 + k] = P[k * 16 + m];
    // smdO_T layout: smdO_T[n=d, k=i] = dO[i, d]  (transposed)
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k)
            dOt[n * 16 + k] = dO[k * 8 + n];

    // CPU reference: D = P^T · dO  (D shape [m=16, n=8])
    float D_cpu[16 * 8];
    cpu_pt_dO(P, dO, D_cpu);

    // Cast to FP16
    __half Pt_h[16 * 16], dO_h[16 * 8], dOt_h[8 * 16];
    for (int i = 0; i < 256; ++i) Pt_h[i] = __float2half_rn(Pt[i]);
    for (int i = 0; i < 128; ++i) dO_h[i] = __float2half_rn(dO[i]);
    for (int i = 0; i < 128; ++i) dOt_h[i] = __float2half_rn(dOt[i]);

    // Upload to GPU
    __half *d_Pt = nullptr, *d_dO = nullptr, *d_dOt = nullptr;
    float  *d_D_B1 = nullptr, *d_D_B2 = nullptr;
    CK(cudaMalloc(&d_Pt,  256 * sizeof(__half)));
    CK(cudaMalloc(&d_dO,  128 * sizeof(__half)));
    CK(cudaMalloc(&d_dOt, 128 * sizeof(__half)));
    CK(cudaMalloc(&d_D_B1, 32 * 4 * sizeof(float)));
    CK(cudaMalloc(&d_D_B2, 32 * 4 * sizeof(float)));
    CK(cudaMemcpy(d_Pt,  Pt_h,  256 * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dO,  dO_h,  128 * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dOt, dOt_h, 128 * sizeof(__half), cudaMemcpyHostToDevice));

    probe_kernel<<<1, 32>>>(d_Pt, d_dO, d_dOt, d_D_B1, d_D_B2);
    CK(cudaDeviceSynchronize());

    float D_B1[32 * 4], D_B2[32 * 4];
    CK(cudaMemcpy(D_B1, d_D_B1, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(D_B2, d_D_B2, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("=== Probe m16n8k16.row.col.f32.f16.f16.f32 layout ===\n");
    printf("Reference D = P^T·dO (16×8 fp32), |D|_max ≈ %.3f\n",
           [&]{ float m=0; for(int i=0;i<128;++i) if(std::fabs(D_cpu[i])>m) m=std::fabs(D_cpu[i]); return m; }());

    int err_B1 = verify_path("Path B1 (smdO row-major, 2×LDS+pack)", D_B1, D_cpu);
    int err_B2 = verify_path("Path B2 (smdO_T row-major, 1×LDS)   ", D_B2, D_cpu);

    printf("\nVERDICT: ");
    if (err_B1 == 0 && err_B2 == 0) {
        printf("BOTH paths CORRECT. Choice by perf.\n");
        printf("  → Path β feasible: smdO row-major, 2× LDS/pack overhead in hot-loop.\n");
        printf("  → Path α feasible: smdO_T transpose pass + 1× LDS in hot-loop.\n");
        return 0;
    } else if (err_B1 == 0 && err_B2 != 0) {
        printf("ONLY B1 CORRECT. Docs lied on B2; path β forced.\n");
        return 0;
    } else if (err_B1 != 0 && err_B2 == 0) {
        printf("ONLY B2 CORRECT. Docs lied on B1; path α forced.\n");
        return 0;
    } else {
        printf("BOTH PATHS FAIL. Fragment layout sm_120a divergent from docs. INVESTIGATE.\n");
        return 1;
    }
}
