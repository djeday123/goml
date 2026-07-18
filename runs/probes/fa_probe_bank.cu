// fa_probe_bank.cu — калибровка счётчика bank-конфликтов (шаг 2-probe + ext).
//   1 блок, 128 threads (4 warp), 8KB SMEM, N iters.
//   Восемь паттернов адресации, каждый = отдельная template-инстанция → отдельный SASS.
//   PTX-inline ld.shared.u32 / st.shared.u8 volatile — 1 LDS.32 или 1 STS.U8 на итерацию.
//
// Address formulas (per lane, l_div4 = lane>>2, l_mod4 = lane&3):
//   P1: LDS addr = lane * 4                    (linear, 0 conflict эталон)
//   P2: LDS addr = lane * 8                    (2-way эталон)
//   P3: LDS addr = lane * 128                  (32-way эталон, all bank 0)
//   P4: LDS addr = l_div4*68 + l_mod4*4        (dk_new B-load @ STRIDE=68)
//   P5: LDS addr = l_div4*80 + l_mod4*4        (dk_new B-load @ STRIDE=80)
//   P6: STS.U8 addr = l_mod4*4*STRIDE + l_div4 (dk_new Q_T scatter, STRIDE=68/80 через under-P)
//   P7: LDS addr = l_div4*64 + l_mod4*4        (dk_new A-load dS_T @ Br=64)
//   P8: LDS addr = l_div4*68 + ((l_mod4*4) ^ ((l_div4>>1)&3)<<4)  (B-load @68 + XOR swz)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

constexpr int SMEM_BYTES = 16384;  // 16KB для расширенных паттернов P10-P17 (max byte_addr ~10K)

template<int P>
__global__ void probe_kernel(uint32_t *out, int N) {
    __shared__ uint32_t smem[SMEM_BYTES / 4];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // Init smem (uint32 words), each thread its slice
    for (int i = tid; i < SMEM_BYTES / 4; i += blockDim.x) {
        smem[i] = i;
    }
    __syncthreads();

    // Per-pattern byte address
    int byte_addr;
    if constexpr (P == 1) byte_addr = lane * 4;
    else if constexpr (P == 2) byte_addr = lane * 8;
    else if constexpr (P == 3) byte_addr = lane * 128;
    else if constexpr (P == 4) byte_addr = l_div4 * 68 + l_mod4 * 4;
    else if constexpr (P == 5) byte_addr = l_div4 * 80 + l_mod4 * 4;
    // P6a/P6b: STS.U8 Q_T scatter (STRIDE 68 / 80). row = l_mod4, col = l_div4
    else if constexpr (P == 6) byte_addr = l_mod4 * 4 * 68 + l_div4;   // Q_T scatter @68
    else if constexpr (P == 7) byte_addr = l_mod4 * 4 * 80 + l_div4;   // Q_T scatter @80
    // P8: A-load @Br=64
    else if constexpr (P == 8) byte_addr = l_div4 * 64 + l_mod4 * 4;
    // P9: B-load @68 с XOR swz ((l_div4>>1)&3)<<4 byte-shift
    else if constexpr (P == 9) byte_addr = l_div4 * 68 + ((l_mod4 * 4) ^ (((l_div4 >> 1) & 3) << 4));
    // P10..P13: B-load @68 with π on row, ni ∈ {0, 1, 2, 3}
    //   π(r) = 32*((r>>3)&3) + ((r>>5)&3) + 4*(r&7)   — bit-rearrangement
    //   n_d = ni*8 + l_div4
    else if constexpr (P >= 10 && P <= 13) {
        int ni_val = P - 10;
        int n_d = ni_val * 8 + l_div4;
        int pi_n_d = 32 * ((n_d >> 3) & 3) + ((n_d >> 5) & 3) + 4 * (n_d & 7);
        byte_addr = pi_n_d * 68 + l_mod4 * 4;
    }
    // P14/P15: STS.U8 Q_T-scatter @68 with π on row. ks=0, bt=0.
    //   k_lo (P14): k = l_mod4*4;  k_hi (P15): k = l_mod4*4 + 16
    else if constexpr (P == 14 || P == 15) {
        int k_offset = (P == 15) ? 16 : 0;
        int k = l_mod4 * 4 + k_offset;
        int pi_k = 32 * ((k >> 3) & 3) + ((k >> 5) & 3) + 4 * (k & 7);
        byte_addr = pi_k * 68 + l_div4;   // wid=0 fixed
    }
    // P16: packed-STS.32 @68 no π. 16 different rows per warp inst.
    //   Distribute 32 lanes over 16 rows × 2 words per row = 32 targets
    //   row = lane / 2, col_word = lane & 1
    else if constexpr (P == 16) {
        int row = lane >> 1;
        int col_word = lane & 1;
        byte_addr = row * 68 + col_word * 4;
    }
    // P17: packed-STS.32 @68 with π.
    else if constexpr (P == 17) {
        int row = lane >> 1;
        int pi_row = 32 * ((row >> 3) & 3) + ((row >> 5) & 3) + 4 * (row & 7);
        int col_word = lane & 1;
        byte_addr = pi_row * 68 + col_word * 4;
    }
    // P18-P21: B-load @68 with π_V, ni ∈ {0, 1, 2, 3}
    //   π_V(r) = ((r&7)<<2) | (((r>>3)&1)<<1) | ((r>>4)&1) | (r&0x60)
    //   bit-permutation r6 r5 r2 r1 r0 r3 r4
    else if constexpr (P >= 18 && P <= 21) {
        int ni_val = P - 18;
        int n_d = ni_val * 8 + l_div4;
        int pi_v = ((n_d & 7) << 2) | (((n_d >> 3) & 1) << 1) | ((n_d >> 4) & 1) | (n_d & 0x60);
        byte_addr = pi_v * 68 + l_mod4 * 4;
    }
    // P22: STS.U8 Q_T-scatter @68 with π_V, k_lo variant (ks=0, bt=0, hi=0)
    //   row = 4*l_mod4, byte_addr = π_V(row)*68 + l_div4 (wid=0)
    else if constexpr (P == 22) {
        int row = 4 * l_mod4;   // ks=0, bt=0, hi=0
        int pi_v = ((row & 7) << 2) | (((row >> 3) & 1) << 1) | ((row >> 4) & 1) | (row & 0x60);
        byte_addr = pi_v * 68 + l_div4;   // wid=0
    }
    // P23: STS.U8 Q_T-scatter @68 with π_V, k_hi + m_hi variant (ks=0, bt=0, hi=1, m_hi=l_div4+8)
    else if constexpr (P == 23) {
        int row = 16 + 4 * l_mod4;   // ks=0, bt=0, hi=1
        int pi_v = ((row & 7) << 2) | (((row >> 3) & 1) << 1) | ((row >> 4) & 1) | (row & 0x60);
        byte_addr = pi_v * 68 + l_div4 + 8;   // wid=0, m_hi
    }
    // P24: packed-STS.32 @68 with π_V, hk=0 (dk_new pack scatter, ks=0, s0=0, wid=0)
    //   lane: c = l_mod4, p = (l_div4)&3, h = (l_div4)>>2
    //   row  = 0*32 + 16*hk + 4c + p  = 4c + p  (hk=0)
    //   col  = 0*16 + 8*0 + 4*h = 4h
    else if constexpr (P == 24) {
        int c = l_mod4;
        int p = (l_div4) & 3;
        int h = (l_div4) >> 2;
        int hk = 0;
        int row = 16*hk + 4*c + p;
        int pi_v = ((row & 7) << 2) | (((row >> 3) & 1) << 1) | ((row >> 4) & 1) | (row & 0x60);
        byte_addr = pi_v * 68 + 4*h;   // wid=0, s0=0
    }
    // P25: packed-STS.32 @68 with π_V, hk=1
    else if constexpr (P == 25) {
        int c = l_mod4;
        int p = (l_div4) & 3;
        int h = (l_div4) >> 2;
        int hk = 1;
        int row = 16*hk + 4*c + p;
        int pi_v = ((row & 7) << 2) | (((row >> 3) & 1) << 1) | ((row >> 4) & 1) | (row & 0x60);
        byte_addr = pi_v * 68 + 4*h;
    }
    else byte_addr = 0;

    // Cvt generic-to-shared → uint32 sm_addr
    uint32_t sm_addr = __cvta_generic_to_shared(&smem[0]);
    sm_addr += (uint32_t)byte_addr;

    // Anti-hoist trick: add same iteration-dependent offset to ALL lanes.
    //   Shifts entire warp uniformly → intra-warp bank pattern preserved,
    //   но compiler не может hoist LDS (address changes per iter).
    uint32_t acc = 0;
    for (int i = 0; i < N; ++i) {
        uint32_t sm_addr_iter = sm_addr + (uint32_t)((i & 0x0F) * 128);   // 0..1920
        if constexpr (P == 6 || P == 7 || P == 14 || P == 15 || P == 22 || P == 23) {
            // STS.U8 side-effect
            uint32_t data = (uint32_t)i;
            asm volatile("st.shared.u8 [%0], %1;" :: "r"(sm_addr_iter), "r"(data));
        } else if constexpr (P == 16 || P == 17 || P == 24 || P == 25) {
            // STS.32 (4-byte store)
            uint32_t data = (uint32_t)i;
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(sm_addr_iter), "r"(data));
        } else {
            uint32_t val;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(sm_addr_iter));
            acc ^= val;
        }
    }

    // For STS-patterns, __syncthreads is enough anti-DCE (side effect visible cross-warp)
    if constexpr (P == 6 || P == 7 || P == 14 || P == 15 || P == 16 || P == 17 || P == 22 || P == 23 || P == 24 || P == 25) {
        __syncthreads();
        // Read aligned byte for sink (l_div4=0 lanes only)
        if (lane == 0) {
            uint32_t val;
            uint32_t sm_addr_read = __cvta_generic_to_shared(&smem[0]);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(sm_addr_read));
            acc = val;
        }
    }

    // Anti-DCE: single writeback
    out[tid] = acc;
}

// Explicit instantiations — 9 отдельных kernel entries в SASS
template __global__ void probe_kernel<1>(uint32_t*, int);
template __global__ void probe_kernel<2>(uint32_t*, int);
template __global__ void probe_kernel<3>(uint32_t*, int);
template __global__ void probe_kernel<4>(uint32_t*, int);
template __global__ void probe_kernel<5>(uint32_t*, int);
template __global__ void probe_kernel<6>(uint32_t*, int);
template __global__ void probe_kernel<7>(uint32_t*, int);
template __global__ void probe_kernel<8>(uint32_t*, int);
template __global__ void probe_kernel<9>(uint32_t*, int);
template __global__ void probe_kernel<10>(uint32_t*, int);
template __global__ void probe_kernel<11>(uint32_t*, int);
template __global__ void probe_kernel<12>(uint32_t*, int);
template __global__ void probe_kernel<13>(uint32_t*, int);
template __global__ void probe_kernel<14>(uint32_t*, int);
template __global__ void probe_kernel<15>(uint32_t*, int);
template __global__ void probe_kernel<16>(uint32_t*, int);
template __global__ void probe_kernel<17>(uint32_t*, int);
template __global__ void probe_kernel<18>(uint32_t*, int);
template __global__ void probe_kernel<19>(uint32_t*, int);
template __global__ void probe_kernel<20>(uint32_t*, int);
template __global__ void probe_kernel<21>(uint32_t*, int);
template __global__ void probe_kernel<22>(uint32_t*, int);
template __global__ void probe_kernel<23>(uint32_t*, int);
template __global__ void probe_kernel<24>(uint32_t*, int);
template __global__ void probe_kernel<25>(uint32_t*, int);

int main(int argc, char **argv) {
    int P = (argc >= 2) ? std::atoi(argv[1]) : 4;
    int N = (argc >= 3) ? std::atoi(argv[2]) : 100000;

    uint32_t *dout;
    cudaMalloc(&dout, 128 * sizeof(uint32_t));

    // Fingerprint gate (проверка SASS-целостности)
    cudaFuncAttributes fa;
    void *fptr = nullptr;
    switch (P) {
        case 1: fptr = (void*)probe_kernel<1>; break;
        case 2: fptr = (void*)probe_kernel<2>; break;
        case 3: fptr = (void*)probe_kernel<3>; break;
        case 4: fptr = (void*)probe_kernel<4>; break;
        case 5: fptr = (void*)probe_kernel<5>; break;
        case 6: fptr = (void*)probe_kernel<6>; break;
        case 7: fptr = (void*)probe_kernel<7>; break;
        case 8: fptr = (void*)probe_kernel<8>; break;
        case 9: fptr = (void*)probe_kernel<9>; break;
        case 10: fptr = (void*)probe_kernel<10>; break;
        case 11: fptr = (void*)probe_kernel<11>; break;
        case 12: fptr = (void*)probe_kernel<12>; break;
        case 13: fptr = (void*)probe_kernel<13>; break;
        case 14: fptr = (void*)probe_kernel<14>; break;
        case 15: fptr = (void*)probe_kernel<15>; break;
        case 16: fptr = (void*)probe_kernel<16>; break;
        case 17: fptr = (void*)probe_kernel<17>; break;
        case 18: fptr = (void*)probe_kernel<18>; break;
        case 19: fptr = (void*)probe_kernel<19>; break;
        case 20: fptr = (void*)probe_kernel<20>; break;
        case 21: fptr = (void*)probe_kernel<21>; break;
        case 22: fptr = (void*)probe_kernel<22>; break;
        case 23: fptr = (void*)probe_kernel<23>; break;
        case 24: fptr = (void*)probe_kernel<24>; break;
        case 25: fptr = (void*)probe_kernel<25>; break;
        default: fprintf(stderr, "invalid pattern %d\n", P); return 1;
    }
    if (cudaFuncGetAttributes(&fa, fptr) == cudaSuccess) {
        printf("probe P%d: FINGERPRINT numRegs=%d, sharedSizeBytes=%zu, "
               "maxThreadsPerBlock=%d, N=%d\n",
               P, fa.numRegs, fa.sharedSizeBytes, fa.maxThreadsPerBlock, N);
    }

    // Launch
    switch (P) {
        case 1: probe_kernel<1><<<1, 128>>>(dout, N); break;
        case 2: probe_kernel<2><<<1, 128>>>(dout, N); break;
        case 3: probe_kernel<3><<<1, 128>>>(dout, N); break;
        case 4: probe_kernel<4><<<1, 128>>>(dout, N); break;
        case 5: probe_kernel<5><<<1, 128>>>(dout, N); break;
        case 6: probe_kernel<6><<<1, 128>>>(dout, N); break;
        case 7: probe_kernel<7><<<1, 128>>>(dout, N); break;
        case 8: probe_kernel<8><<<1, 128>>>(dout, N); break;
        case 9: probe_kernel<9><<<1, 128>>>(dout, N); break;
        case 10: probe_kernel<10><<<1, 128>>>(dout, N); break;
        case 11: probe_kernel<11><<<1, 128>>>(dout, N); break;
        case 12: probe_kernel<12><<<1, 128>>>(dout, N); break;
        case 13: probe_kernel<13><<<1, 128>>>(dout, N); break;
        case 14: probe_kernel<14><<<1, 128>>>(dout, N); break;
        case 15: probe_kernel<15><<<1, 128>>>(dout, N); break;
        case 16: probe_kernel<16><<<1, 128>>>(dout, N); break;
        case 17: probe_kernel<17><<<1, 128>>>(dout, N); break;
        case 18: probe_kernel<18><<<1, 128>>>(dout, N); break;
        case 19: probe_kernel<19><<<1, 128>>>(dout, N); break;
        case 20: probe_kernel<20><<<1, 128>>>(dout, N); break;
        case 21: probe_kernel<21><<<1, 128>>>(dout, N); break;
        case 22: probe_kernel<22><<<1, 128>>>(dout, N); break;
        case 23: probe_kernel<23><<<1, 128>>>(dout, N); break;
        case 24: probe_kernel<24><<<1, 128>>>(dout, N); break;
        case 25: probe_kernel<25><<<1, 128>>>(dout, N); break;
    }
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "kernel launch err: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // Sink acc to prevent full elimination
    uint32_t hout[128];
    cudaMemcpy(hout, dout, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t sink = 0;
    for (int i = 0; i < 128; ++i) sink ^= hout[i];
    printf("probe P%d: N=%d sink=%u (anti-DCE)\n", P, N, sink);

    cudaFree(dout);
    return 0;
}
