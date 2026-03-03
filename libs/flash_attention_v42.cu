// =============================================================================
// FlashAttention v42 — Transposed V in SMEM + ldmatrix.x4 (SM89)
// =============================================================================
// Based on v20 (151T, 92%). Changes:
//   1. V stored transposed in SMEM: V_t[d][k] instead of V[k][d]
//      This allows ldmatrix.x4 (no .trans) for V fragments in PV loop.
//      64 ldm2t -> 32 ldm4 in PV = 50% fewer V load instructions.
//   2. K still uses ldm4 pairs from v41 (ld_b2_sw).
//   3. Transposed V load function: load_tile_transpose()
//
// SMEM layout:
//   buf0/buf1 each 16KB. Used as:
//     K: 64 rows x 128 cols, stride=128  (same as v20)
//     V_t: 128 rows x 64 cols, stride=64 (transposed)
//   Same total size, different interpretation.
// =============================================================================

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FA42_BR 64
#define FA42_BC 64
#define FA42_D 128
#define FA42_THREADS 128
#define FA42_K_STRIDE 128 // K: 64 rows x 128 cols
#define FA42_VT_STRIDE 64 // V_t: 128 rows x 64 cols

// Buffer size: max(BC*K_STRIDE, D*VT_STRIDE) * sizeof(half)
// = max(64*128, 128*64) * 2 = 16384 bytes = 16KB each
#define FA42_BUF_SIZE (FA42_BC * FA42_K_STRIDE * (int)sizeof(__half))

__device__ __forceinline__ int swz(int row, int col)
{
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}
__device__ __forceinline__ void cpa16(void *s, const void *g, int n)
{
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldm4(uint32_t &r0, uint32_t &r1,
                                     uint32_t &r2, uint32_t &r3, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm2t(uint32_t &r0, uint32_t &r1, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16.trans {%0,%1},[%2];"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma16816(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// A from swizzled SMEM (Q) — ldmatrix.x4
__device__ __forceinline__ void ld_a_sw(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const __half *sm, int stride, int rb, int kb, int lane)
{
    int sub = lane / 8, sr = lane % 8;
    int r = rb + (sub & 1) * 8 + sr, lc = kb + (sub >> 1) * 8;
    ldm4(a0, a1, a2, a3, &sm[r * stride + swz(r, lc)]);
}

// B from swizzled SMEM (K) — ldmatrix.x4, TWO n-tiles
__device__ __forceinline__ void ld_b2_sw(
    uint32_t &b0a, uint32_t &b1a, uint32_t &b0b, uint32_t &b1b,
    const __half *sm, int stride, int nb, int kb, int lane)
{
    int sub = lane / 8;
    int sr = lane % 8;
    int r = nb + (sub >> 1) * 8 + sr;
    int lc = kb + (sub & 1) * 8;
    ldm4(b0a, b1a, b0b, b1b, &sm[r * stride + swz(r, lc)]);
}

// B from transposed V in SMEM — ldmatrix.x4, TWO n-tiles at once
// V_t stored as [D][BC] with stride=BC=64, swizzled.
// For PV MMA: P(m16k16) @ V_t(n8k16)
// B operand needs col-major access: B[k][n]
// V_t[d][k] in SMEM, row-major. For B fragment:
//   n = d dimension (output cols), k = BC dimension (reduction)
// ldmatrix.x4 loads 4 x 8x8 tiles = two B fragments
//   sub 0,1 -> first n-tile (d_base..d_base+7)
//   sub 2,3 -> second n-tile (d_base+8..d_base+15)
__device__ __forceinline__ void ld_vt2_sw(
    uint32_t &b0a, uint32_t &b1a, uint32_t &b0b, uint32_t &b1b,
    const __half *sm, int stride, int nb, int kb, int lane)
{
    // For V_t: rows = d dimension, cols = k dimension
    // Same pattern as ld_b2_sw but on V_t layout
    int sub = lane / 8;
    int sr = lane % 8;
    int r = nb + (sub >> 1) * 8 + sr; // d dimension
    int lc = kb + (sub & 1) * 8;      // k dimension
    ldm4(b0a, b1a, b0b, b1b, &sm[r * stride + swz(r, lc)]);
}

// Standard tile load (K, Q): row-major [rows x 128], stride=128
__device__ __forceinline__ void load_tile(
    __half *dst, const __half *src, int start, int rows,
    int seq_len, int head_dim, int stride)
{
    int cols_per_row = head_dim / 8; // 16 for D=128
    int total = rows * cols_per_row;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA42_THREADS)
    {
        int row = c / cols_per_row, lch = c % cols_per_row, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * stride + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// Transposed tile load for V: V[k][d] in global -> V_t[d][k] in SMEM
// V_t layout: [D][BC] with stride=BC, swizzled
// Global V is row-major: V[k * head_dim + d]
// We need to write V_t[d * stride + swz(d, k_chunk)]
// cp.async copies 16 bytes = 8 halves from contiguous global memory
// Global V: V[k][d..d+7] is contiguous (row-major, along d)
// But we want V_t[d][k] — different layout!
//
// Strategy: each thread loads 8 contiguous halves from V[k][d..d+7],
// but writes them to V_t[d..d+7][k] which is NOT contiguous in SMEM.
// cp.async requires contiguous src AND contiguous dst.
//
// Alternative: load V[k][d..d+7] contiguous -> write to scratch,
// then transpose. But that's complex.
//
// Better: use cp.async along the K dimension.
// V[k..k+7][d] — NOT contiguous in global (stride = head_dim).
//
// Simplest approach: load V normally into SMEM, then do in-place transpose.
// But that requires extra sync and is wasteful.
//
// BEST approach: Load V as [BC][D] into SMEM normally with stride=D=128.
// Then use ldm2t for V (like v20). BUT we wanted to avoid ldm2t!
//
// Actually, let's reconsider. gau-nernst on 5090 stores V row-major too,
// and uses ldmatrix.x2.trans for V in v1-v3. In v4 he switches to
// ldmatrix.x4 for BOTH K and V. For V he can do this because:
//   - V acts as B in MMA: P(m16k16) @ V -> O(m16n8)
//   - B needs n8k16 fragment
//   - With V stored as V[k][d] row-major:
//     * ldm2t reads V[k][d] and transposes: gives B[n=d][k] layout
//     * ldm4 (no trans) on V[k][d] gives B[k][n=d] which is WRONG
//   - With V_t[d][k] stored transposed:
//     * ldm4 reads rows along k dimension -> correct B[n=d][k] layout? NO!
//
// Let me re-think the fragment layout more carefully.
// mma.m16n8k16: A[m][k] row-major, B[k][n] col-major
// B fragment via ldmatrix.x2: loads 2x 8x8 from rows of SMEM.
//   The rows of SMEM map to n-dimension of B.
//   Within each row, 16 elements map to k-dimension.
//   So SMEM layout for B is: B_smem[n][k], row-major.
//
// For QK^T: B = K^T. K stored as K[n][k] where n=kv_seq, k=head_dim.
//   K_smem[n][k] = K[kv_pos][head_dim] — already correct!
//   ldmatrix reads K_smem rows -> n dimension. Correct.
//
// For PV: B = V. We need B[k][n] col-major, or equivalently B_smem[n][k] row-major.
//   n = head_dim, k = kv_seq.
//   So V_smem should be V_smem[d][kv] = V_t[d][k], row-major.
//   Then ldmatrix reads rows -> d dimension (= n dimension of B). Correct!
//   And within row: k dimension. Correct!
//
// So yes: store V transposed as V_t[d][k] in SMEM, then ldmatrix.x4 works!
//
// The challenge is the transpose during load. cp.async needs contiguous src.
// V in global is V[k][d], row-major. Contiguous along d.
// V_t in SMEM is V_t[d][k], row-major with stride=BC. Contiguous along k.
//
// We need to write V_t[d][k] = V[k][d]. For cp.async 16 bytes:
// Option A: read V[k][d..d+7] (contiguous in global), write to 8 different
//   SMEM locations V_t[d][k], V_t[d+1][k], ... — NOT contiguous. Can't use cp.async.
// Option B: read V[k..k+7][d] — NOT contiguous in global (stride=head_dim).
//   Can't use cp.async.
//
// cp.async CANNOT do transpose. We must load V normally then transpose in SMEM,
// OR load V normally and use ldm2t (current approach).
//
// So gau-nernst must have found another way. Let me re-read his code...
// He likely loads V normally and uses a SMEM transpose kernel between
// the V load and the PV computation. Or he uses a different trick entirely.
//
// Actually, re-reading his blog: "V will require transposed ldmatrix"
// And in v4: "ldmatrix.x4 for K and V" — he uses ldmatrix.x4 for BOTH.
// For K: groups 2 n-tiles into one x4 call.
// For V: he might group 2 n-tiles into one x4.trans call? But .x4.trans
// doesn't exist... OR he loads V row-major and uses x4 differently.
//
// Wait — ldmatrix.x4.trans DOES exist on some architectures!
// PTX spec: "ldmatrix.sync.aligned.x1.m8n8.shared.b16.trans"
// The .trans modifier is available for x1, x2, x4 on sm_75+.
// Let me check: SM89 PTX ISA...
// Actually looking at PTX 8.x: ldmatrix supports .trans with .x1, .x2, .x4
// It's .x4.trans that I assumed didn't exist — but it DOES!
//
// From PTX ISA docs: "ldmatrix.sync.aligned.num.trans.shape.shared.b16"
// num = .x1, .x2, .x4. trans is optional modifier.
// So ldmatrix.x4.trans IS valid!
//
// This changes everything. We can use ldm4t for V and get 2 n-tiles at once!
// =============================================================================

// ldmatrix.x4.trans — DOES exist on SM75+!
__device__ __forceinline__ void ldm4t(uint32_t &r0, uint32_t &r1,
                                      uint32_t &r2, uint32_t &r3, const void *p)
{
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16.trans {%0,%1,%2,%3},[%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// B from swizzled SMEM transposed (V) — ldmatrix.x4.trans, TWO n-tiles
// V stored as V[k][d] row-major with stride=128, swizzled.
// For PV: B fragment needs n=d, k=kv_seq reduction dim.
// ldmatrix.x4.trans loads 4 x 8x8 tiles with transpose.
// sub 0,1 -> first n-tile (d_base..d_base+7)
// sub 2,3 -> second n-tile (d_base+8..d_base+15)
// Each sub: thread row addr -> k dimension, transpose gives n=d output.
__device__ __forceinline__ void ld_vt2_sw_trans(
    uint32_t &b0a, uint32_t &b1a, uint32_t &b0b, uint32_t &b1b,
    const __half *sm, int stride, int nb, int kb, int lane)
{
    // nb = d_base (n-tile base in d dimension, groups of 16)
    // kb = k_base (k-step in BC dimension)
    // For .trans: thread addresses rows in k dimension
    // sub 0: k[kb..kb+7],   reads from col nb      -> transposes to n=nb..nb+7
    // sub 1: k[kb+8..kb+15], reads from col nb      -> transposes to n=nb..nb+7  (second k half)
    // sub 2: k[kb..kb+7],   reads from col nb+8    -> transposes to n=nb+8..nb+15
    // sub 3: k[kb+8..kb+15], reads from col nb+8   -> transposes to n=nb+8..nb+15
    int sub = lane / 8;
    int sr = lane % 8;
    int k = kb + (sub & 1) * 8 + sr; // k dimension (row in V_smem)
    int d = nb + (sub >> 1) * 8;     // d dimension (col in V_smem, base of 8x8 tile)
    ldm4t(b0a, b1a, b0b, b1b, &sm[k * stride + swz(k, d)]);
}

// Standard tile load with configurable stride
__device__ __forceinline__ void load_tile_k(
    __half *dst, const __half *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CPR = 16; // 128 / 8
    int total = rows * CPR;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA42_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA42_K_STRIDE + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// V load: same as K (row-major V[k][d], stride=128)
// V is stored identically to K in SMEM — read via ldm4t
__device__ __forceinline__ void load_tile_v(
    __half *dst, const __half *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CPR = 16;
    int total = rows * CPR;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA42_THREADS)
    {
        int row = c / CPR, lch = c % CPR, gr = start + row;
        int pch = lch ^ (row & 7);
        cpa16(&dst[row * FA42_K_STRIDE + pch * 8],
              &src[gr * head_dim + lch * 8], (gr < seq_len) ? 16 : 0);
    }
}

// =============================================================================
__global__ void __launch_bounds__(FA42_THREADS, 2)
    flash_attention_v42_kernel(
        const __half *__restrict__ Q, const __half *__restrict__ K,
        const __half *__restrict__ V, __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale)
{
    int nqt = (seq_len + FA42_BR - 1) / FA42_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA42_BR;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane >> 2, tid = lane & 3;

    extern __shared__ char raw[];
    __half *buf0 = (__half *)raw;
    __half *buf1 = (__half *)(raw + FA42_BUF_SIZE);

    int hs = seq_len * head_dim;
    const __half *Qh = Q + bh * hs, *Kh = K + bh * hs, *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    // Load Q -> registers
    load_tile_k(buf0, Qh, qs, FA42_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    int mrb = wid * 16;
    uint32_t Qr[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
        ld_a_sw(Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                buf0, FA42_K_STRIDE, mrb, ks * 16, lane);
    __syncthreads();

    float Or[16][4];
#pragma unroll
    for (int t = 0; t < 16; t++)
        Or[t][0] = Or[t][1] = Or[t][2] = Or[t][3] = 0;

    float rmax[2] = {-1e30f, -1e30f}, rsexp[2] = {0, 0};
    int nkv = (seq_len + FA42_BC - 1) / FA42_BC;

    // Prefetch K[0]
    load_tile_k(buf0, Kh, 0, FA42_BC, seq_len, head_dim);
    cpa_commit();

    for (int kv = 0; kv < nkv; kv++)
    {
        int kvs = kv * FA42_BC;
        if (causal && kvs > qs + FA42_BR - 1)
            break;

        __half *cur = (kv & 1) ? buf1 : buf0;
        __half *nxt = (kv & 1) ? buf0 : buf1;

        cpa_wait<0>();
        __syncthreads();

        // ==== QK^T: ldm4 pairs (32 ldm4) ====
        float Sr[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
            Sr[nt][0] = Sr[nt][1] = Sr[nt][2] = Sr[nt][3] = 0;

#pragma unroll
        for (int ks = 0; ks < 8; ks++)
        {
#pragma unroll
            for (int np = 0; np < 4; np++)
            {
                uint32_t b0a, b1a, b0b, b1b;
                ld_b2_sw(b0a, b1a, b0b, b1b, cur, FA42_K_STRIDE, np * 16, ks * 16, lane);
                mma16816(Sr[np * 2][0], Sr[np * 2][1], Sr[np * 2][2], Sr[np * 2][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0a, b1a, Sr[np * 2][0], Sr[np * 2][1], Sr[np * 2][2], Sr[np * 2][3]);
                mma16816(Sr[np * 2 + 1][0], Sr[np * 2 + 1][1], Sr[np * 2 + 1][2], Sr[np * 2 + 1][3],
                         Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                         b0b, b1b, Sr[np * 2 + 1][0], Sr[np * 2 + 1][1], Sr[np * 2 + 1][2], Sr[np * 2 + 1][3]);
            }
        }

        // K consumed — load V[kv] into cur, K[kv+1] into nxt
        __syncthreads();
        load_tile_v(cur, Vh, kvs, FA42_BC, seq_len, head_dim);
        cpa_commit();
        int nkvs = (kv + 1) * FA42_BC;
        bool has_nxt = (kv + 1 < nkv) && (!causal || nkvs <= qs + FA42_BR - 1);
        if (has_nxt)
            load_tile_k(nxt, Kh, nkvs, FA42_BC, seq_len, head_dim);
        cpa_commit();

        // Scale + causal mask
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            Sr[nt][0] *= scale;
            Sr[nt][1] *= scale;
            Sr[nt][2] *= scale;
            Sr[nt][3] *= scale;
            if (causal)
            {
                int gq0 = qs + mrb + gid, gq8 = gq0 + 8;
                int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;
                if (gk0 > gq0)
                    Sr[nt][0] = -1e30f;
                if (gk1 > gq0)
                    Sr[nt][1] = -1e30f;
                if (gk0 > gq8)
                    Sr[nt][2] = -1e30f;
                if (gk1 > gq8)
                    Sr[nt][3] = -1e30f;
                if (gq0 >= seq_len)
                {
                    Sr[nt][0] = -1e30f;
                    Sr[nt][1] = -1e30f;
                }
                if (gq8 >= seq_len)
                {
                    Sr[nt][2] = -1e30f;
                    Sr[nt][3] = -1e30f;
                }
                if (gk0 >= seq_len)
                {
                    Sr[nt][0] = -1e30f;
                    Sr[nt][2] = -1e30f;
                }
                if (gk1 >= seq_len)
                {
                    Sr[nt][1] = -1e30f;
                    Sr[nt][3] = -1e30f;
                }
            }
        }

        // Online softmax
        float nm[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            nm[0] = fmaxf(nm[0], fmaxf(Sr[nt][0], Sr[nt][1]));
            nm[1] = fmaxf(nm[1], fmaxf(Sr[nt][2], Sr[nt][3]));
        }
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 1));
        nm[0] = fmaxf(nm[0], __shfl_xor_sync(0xffffffff, nm[0], 2));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 1));
        nm[1] = fmaxf(nm[1], __shfl_xor_sync(0xffffffff, nm[1], 2));
        nm[0] = fmaxf(nm[0], rmax[0]);
        nm[1] = fmaxf(nm[1], rmax[1]);

        float rsc0 = __expf(rmax[0] - nm[0]), rsc1 = __expf(rmax[1] - nm[1]);
#pragma unroll
        for (int t = 0; t < 16; t++)
        {
            Or[t][0] *= rsc0;
            Or[t][1] *= rsc0;
            Or[t][2] *= rsc1;
            Or[t][3] *= rsc1;
        }
        rmax[0] = nm[0];
        rmax[1] = nm[1];

        float ns[2] = {0, 0};
        uint32_t Pr[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            Sr[nt][0] = __expf(Sr[nt][0] - rmax[0]);
            Sr[nt][1] = __expf(Sr[nt][1] - rmax[0]);
            Sr[nt][2] = __expf(Sr[nt][2] - rmax[1]);
            Sr[nt][3] = __expf(Sr[nt][3] - rmax[1]);
            ns[0] += Sr[nt][0] + Sr[nt][1];
            ns[1] += Sr[nt][2] + Sr[nt][3];
            int pi = nt / 2, half = nt % 2;
            __half2 *p = (__half2 *)Pr[pi];
            p[half * 2] = __halves2half2(__float2half(Sr[nt][0]), __float2half(Sr[nt][1]));
            p[half * 2 + 1] = __halves2half2(__float2half(Sr[nt][2]), __float2half(Sr[nt][3]));
        }
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 1);
        ns[0] += __shfl_xor_sync(0xffffffff, ns[0], 2);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 1);
        ns[1] += __shfl_xor_sync(0xffffffff, ns[1], 2);
        rsexp[0] = rsexp[0] * rsc0 + ns[0];
        rsexp[1] = rsexp[1] * rsc1 + ns[1];

        // ==== PV: O += P @ V via ldm4t pairs (32 ldm4t) ====
        // V in SMEM: V[k][d] row-major, stride=128, swizzled
        // PV: P(m16k16) @ V(n8k16) -> O(m16n8)
        // 4 k-steps (BC/16=4), 16 n-tiles (D/8=16)
        // Group n-tiles into pairs -> 8 x ldm4t per k-step = 32 total
        cpa_wait<1>();
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
#pragma unroll
            for (int np = 0; np < 8; np++)
            {
                uint32_t b0a, b1a, b0b, b1b;
                ld_vt2_sw_trans(b0a, b1a, b0b, b1b,
                                cur, FA42_K_STRIDE, np * 16, ks * 16, lane);

                int nt0 = np * 2, nt1 = np * 2 + 1;
                mma16816(Or[nt0][0], Or[nt0][1], Or[nt0][2], Or[nt0][3],
                         Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3],
                         b0a, b1a, Or[nt0][0], Or[nt0][1], Or[nt0][2], Or[nt0][3]);
                mma16816(Or[nt1][0], Or[nt1][1], Or[nt1][2], Or[nt1][3],
                         Pr[ks][0], Pr[ks][1], Pr[ks][2], Pr[ks][3],
                         b0b, b1b, Or[nt1][0], Or[nt1][1], Or[nt1][2], Or[nt1][3]);
            }
        }
    }

    // Final output
    float li0 = (rsexp[0] > 0) ? 1.f / rsexp[0] : 0, li1 = (rsexp[1] > 0) ? 1.f / rsexp[1] : 0;
    int gr0 = qs + mrb + gid, gr8 = gr0 + 8;
#pragma unroll
    for (int nt = 0; nt < 16; nt++)
    {
        int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
        if (gr0 < seq_len && c0 < head_dim)
            Oh[gr0 * head_dim + c0] = __float2half(Or[nt][0] * li0);
        if (gr0 < seq_len && c1 < head_dim)
            Oh[gr0 * head_dim + c1] = __float2half(Or[nt][1] * li0);
        if (gr8 < seq_len && c0 < head_dim)
            Oh[gr8 * head_dim + c0] = __float2half(Or[nt][2] * li1);
        if (gr8 < seq_len && c1 < head_dim)
            Oh[gr8 * head_dim + c1] = __float2half(Or[nt][3] * li1);
    }
}

// =============================================================================
static int g_FA42_smem = 0;
extern "C"
{
    int flash_attention_v42_forward(
        const void *Q, const void *K, const void *V, void *O,
        int total_heads, int seq_len, int head_dim, int causal, void *stream)
    {
        if (head_dim != 128)
            return -1;
        int smem = 2 * FA42_BUF_SIZE;
        if (smem > g_FA42_smem)
        {
            cudaError_t e = cudaFuncSetAttribute(flash_attention_v42_kernel,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess)
                return (int)e;
            g_FA42_smem = smem;
        }
        float sc = 1.0f / sqrtf((float)head_dim);
        int nqt = (seq_len + FA42_BR - 1) / FA42_BR;
        flash_attention_v42_kernel<<<total_heads * nqt, FA42_THREADS, smem, (cudaStream_t)stream>>>(
            (const __half *)Q, (const __half *)K, (const __half *)V, (__half *)O,
            seq_len, head_dim, causal, sc);
        return (int)cudaGetLastError();
    }
}
