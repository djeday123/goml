// TMA + SWIZZLE_128B isolated probe on sm_120a.
// Measures: (a) bank conflicts on LDS reads of TMA-swizzled buffer,
//           (b) bit-exact correctness vs baseline (cp.async no-swizzle),
//           (c) how to read the swizzled buffer (plain LDS transparent? ldmatrix transparent?).

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} } while(0)

constexpr int STRIDE = 128;
constexpr int ROWS   = 64;
constexpr int SMEM_BYTES = ROWS * STRIDE;   // 8192 bytes
constexpr int NBLOCKS = 16384;              // match dQ production block count for NCu aggregation

// =============================================================================
// Kernel A: TMA + SWIZZLE_128B load, then LDS read (MMA-A K read pattern mirror)
// =============================================================================
__global__ void tma_swizzled_lds(const __grid_constant__ CUtensorMap tm, uint32_t* out_regs) {
    __shared__ __align__(1024) uint8_t sm[SMEM_BYTES];
    __shared__ __align__(8)    uint64_t mbar[1];

    int tid = threadIdx.x;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        // TMA 2D load: full tile (128 x 64) from global at (0,0) → shared
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"((uint32_t)__cvta_generic_to_shared(sm)),
               "l"(&tm),
               "r"(0), "r"(0),
               "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "n"(SMEM_BYTES));
    }
    // Wait for TMA completion
    asm volatile(
        "{.reg .pred p;"
        "waitLoop:"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;"
        "@p bra doneWait;"
        "bra waitLoop;"
        "doneWait:"
        "}"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));

    // LDS reads — dQ MMA-A K read pattern (each lane 4 uint32 = 16 FP8 bytes from row)
    int lane_k = tid >> 2;
    int lane_n = tid & 3;
    int row = lane_k & 7;
    int col_base = lane_n * 4;

    uint32_t r0 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 0]);
    uint32_t r1 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 16]);
    uint32_t r2 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 32]);
    uint32_t r3 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 48]);

    int bid = blockIdx.x;
    out_regs[bid * 128 + tid * 4 + 0] = r0;
    out_regs[bid * 128 + tid * 4 + 1] = r1;
    out_regs[bid * 128 + tid * 4 + 2] = r2;
    out_regs[bid * 128 + tid * 4 + 3] = r3;
}

// =============================================================================
// Kernel B: TMA + SWIZZLE_128B load, then ldmatrix read
// =============================================================================
__global__ void tma_swizzled_ldmatrix(const __grid_constant__ CUtensorMap tm, uint32_t* out_regs) {
    __shared__ __align__(1024) uint8_t sm[SMEM_BYTES];
    __shared__ __align__(8)    uint64_t mbar[1];

    int tid = threadIdx.x;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"((uint32_t)__cvta_generic_to_shared(sm)),
               "l"(&tm),
               "r"(0), "r"(0),
               "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "n"(SMEM_BYTES));
    }
    asm volatile(
        "{.reg .pred p;"
        "waitLoopB:"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;"
        "@p bra doneWaitB;"
        "bra waitLoopB;"
        "doneWaitB:"
        "}"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));

    int row = tid & 7;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm[row * STRIDE]));
    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));

    int bid = blockIdx.x;
    out_regs[bid * 128 + tid * 4 + 0] = r0;
    out_regs[bid * 128 + tid * 4 + 1] = r1;
    out_regs[bid * 128 + tid * 4 + 2] = r2;
    out_regs[bid * 128 + tid * 4 + 3] = r3;
}

// =============================================================================
// Kernel D: TMA + SWIZZLE_128B + swizzle-aware LDS reads
// Physical address formula: phys_col = logical_col XOR ((row & 7) << 4)
// =============================================================================
__global__ void tma_swizzled_lds_aware(const __grid_constant__ CUtensorMap tm, uint32_t* out_regs) {
    __shared__ __align__(1024) uint8_t sm[SMEM_BYTES];
    __shared__ __align__(8)    uint64_t mbar[1];

    int tid = threadIdx.x;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"((uint32_t)__cvta_generic_to_shared(sm)),
               "l"(&tm),
               "r"(0), "r"(0),
               "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "n"(SMEM_BYTES));
    }
    asm volatile(
        "{.reg .pred p;"
        "waitLoopD:"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;"
        "@p bra doneWaitD;"
        "bra waitLoopD;"
        "doneWaitD:"
        "}"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));

    // Same lane→row/col mapping as baseline, but XOR col with (row<<4) for swizzle-aware read
    int lane_k = tid >> 2;
    int lane_n = tid & 3;
    int row = lane_k & 7;
    int col_base = lane_n * 4;
    int xor_val = (row & 7) << 4;

    uint32_t r0 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 0)  ^ xor_val)]);
    uint32_t r1 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 16) ^ xor_val)]);
    uint32_t r2 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 32) ^ xor_val)]);
    uint32_t r3 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 48) ^ xor_val)]);

    int bid = blockIdx.x;
    out_regs[bid * 128 + tid * 4 + 0] = r0;
    out_regs[bid * 128 + tid * 4 + 1] = r1;
    out_regs[bid * 128 + tid * 4 + 2] = r2;
    out_regs[bid * 128 + tid * 4 + 3] = r3;
}

// =============================================================================
// Kernel E: MANUAL XOR — plain cp.async with XOR-swizzled destination +
//           LDS with matching XOR-swizzled source. No TMA involved.
// Same swizzle formula as SWIZZLE_128B: chunk_phys = chunk_logical XOR (row & 7)
// = byte_addr_phys = row*STRIDE + (col_byte XOR ((row & 7) << 4))
// =============================================================================
__global__ void cpasync_xor_lds(const uint8_t* g, uint32_t* out_regs) {
    __shared__ __align__(16) uint8_t sm[SMEM_BYTES];
    int tid = threadIdx.x;

    // cp.async 16-byte writes with XOR-swizzled destination
    for (int chunk = 0; chunk < SMEM_BYTES; chunk += 32 * 16) {
        int offset = chunk + tid * 16;   // logical global offset
        if (offset + 16 <= SMEM_BYTES) {
            int row = offset / STRIDE;
            int col_byte = offset & (STRIDE - 1);
            int xor_val = (row & 7) << 4;
            int swizzled_offset = row * STRIDE + (col_byte ^ xor_val);   // XOR only chunk bits (4..6)
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&sm[swizzled_offset])),
                   "l"(g + offset));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // LDS reads with matching XOR
    int lane_k = tid >> 2;
    int lane_n = tid & 3;
    int row = lane_k & 7;
    int col_base = lane_n * 4;
    int xor_val = (row & 7) << 4;

    uint32_t r0 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 0)  ^ xor_val)]);
    uint32_t r1 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 16) ^ xor_val)]);
    uint32_t r2 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 32) ^ xor_val)]);
    uint32_t r3 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + ((col_base + 48) ^ xor_val)]);

    int bid = blockIdx.x;
    out_regs[bid * 128 + tid * 4 + 0] = r0;
    out_regs[bid * 128 + tid * 4 + 1] = r1;
    out_regs[bid * 128 + tid * 4 + 2] = r2;
    out_regs[bid * 128 + tid * 4 + 3] = r3;
}

// =============================================================================
// Kernel C: BASELINE — plain cp.async + LDS (mirror dQ current pattern, 8-way conflict)
// =============================================================================
__global__ void cpasync_lds_baseline(const uint8_t* g, uint32_t* out_regs) {
    __shared__ __align__(16) uint8_t sm[SMEM_BYTES];
    int tid = threadIdx.x;
    // cp.async 16-byte load: 8192 bytes / (32 lanes × 16) = 16 rounds per lane
    for (int chunk = 0; chunk < SMEM_BYTES; chunk += 32 * 16) {
        int offset = chunk + tid * 16;
        if (offset + 16 <= SMEM_BYTES) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&sm[offset])),
                   "l"(g + offset));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // Same LDS read pattern as Kernel A
    int lane_k = tid >> 2;
    int lane_n = tid & 3;
    int row = lane_k & 7;
    int col_base = lane_n * 4;

    uint32_t r0 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 0]);
    uint32_t r1 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 16]);
    uint32_t r2 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 32]);
    uint32_t r3 = *reinterpret_cast<uint32_t*>(&sm[row * STRIDE + col_base + 48]);

    int bid = blockIdx.x;
    out_regs[bid * 128 + tid * 4 + 0] = r0;
    out_regs[bid * 128 + tid * 4 + 1] = r1;
    out_regs[bid * 128 + tid * 4 + 2] = r2;
    out_regs[bid * 128 + tid * 4 + 3] = r3;
}

// =============================================================================
// Host driver
// =============================================================================
int main() {
    cudaFree(0);

    // Global tensor: 64 rows × 128 cols FP8, UNIQUE per-byte pattern for swizzle detection
    uint8_t host_in[SMEM_BYTES];
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < STRIDE; c++)
            host_in[r * STRIDE + c] = static_cast<uint8_t>((r * 13 + c * 7) & 0xff);

    uint8_t *d_in = nullptr;
    CK(cudaMalloc(&d_in, SMEM_BYTES));
    CK(cudaMemcpy(d_in, host_in, SMEM_BYTES, cudaMemcpyHostToDevice));

    uint32_t *d_out_tma_lds = nullptr, *d_out_tma_lm = nullptr, *d_out_base = nullptr, *d_out_aware = nullptr, *d_out_xor = nullptr;
    CK(cudaMalloc(&d_out_tma_lds, NBLOCKS * 128 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_out_tma_lm,  NBLOCKS * 128 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_out_base,    NBLOCKS * 128 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_out_aware,   NBLOCKS * 128 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_out_xor,     NBLOCKS * 128 * sizeof(uint32_t)));

    // Setup tensor map with SWIZZLE_128B
    CUtensorMap tm;
    uint64_t global_dims[2]    = {STRIDE, ROWS};   // {cols, rows} — inner-first
    uint64_t global_strides[1] = {STRIDE};
    uint32_t box_dims[2]       = {STRIDE, ROWS};   // full tile
    uint32_t elem_strides[2]   = {1, 1};

    CUresult r = cuTensorMapEncodeTiled(
        &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_in,
        global_dims, global_strides, box_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char* estr; cuGetErrorString(r, &estr);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", estr);
        return 1;
    }

    tma_swizzled_lds<<<NBLOCKS, 32>>>(tm, d_out_tma_lds);
    tma_swizzled_ldmatrix<<<NBLOCKS, 32>>>(tm, d_out_tma_lm);
    cpasync_lds_baseline<<<NBLOCKS, 32>>>(d_in, d_out_base);
    tma_swizzled_lds_aware<<<NBLOCKS, 32>>>(tm, d_out_aware);
    cpasync_xor_lds<<<NBLOCKS, 32>>>(d_in, d_out_xor);
    CK(cudaDeviceSynchronize());

    // Read first block outputs (32 lanes × 4 regs)
    uint32_t h_tma_lds[128], h_tma_lm[128], h_base[128];
    CK(cudaMemcpy(h_tma_lds, d_out_tma_lds, sizeof(h_tma_lds), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_tma_lm,  d_out_tma_lm,  sizeof(h_tma_lm),  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_base,    d_out_base,    sizeof(h_base),    cudaMemcpyDeviceToHost));

    printf("Pattern: sm[r][c] = (r<<4)|(c&0xf).  First block, 32 lanes × 4 regs:\n\n");
    printf("%-4s | %-42s | %-42s | %-42s\n",
           "lane", "BASELINE (cp.async+LDS)", "TMA-swizzle+LDS", "TMA-swizzle+ldmatrix");
    for (int lane = 0; lane < 32; lane++) {
        printf("%2d   | %08x %08x %08x %08x  | %08x %08x %08x %08x  | %08x %08x %08x %08x\n",
               lane,
               h_base[lane*4+0], h_base[lane*4+1], h_base[lane*4+2], h_base[lane*4+3],
               h_tma_lds[lane*4+0], h_tma_lds[lane*4+1], h_tma_lds[lane*4+2], h_tma_lds[lane*4+3],
               h_tma_lm[lane*4+0], h_tma_lm[lane*4+1], h_tma_lm[lane*4+2], h_tma_lm[lane*4+3]);
    }

    uint32_t h_aware[128], h_xor[128];
    CK(cudaMemcpy(h_aware, d_out_aware, sizeof(h_aware), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_xor,   d_out_xor,   sizeof(h_xor),   cudaMemcpyDeviceToHost));

    // Correctness: compare
    int lds_match = 0, lm_match_base = 0, aware_match = 0, xor_match_base = 0, xor_match_aware = 0;
    for (int i = 0; i < 128; i++) {
        if (h_tma_lds[i] == h_base[i]) lds_match++;
        if (h_tma_lm[i]  == h_base[i]) lm_match_base++;
        if (h_aware[i]   == h_base[i]) aware_match++;
        if (h_xor[i]     == h_base[i])  xor_match_base++;
        if (h_xor[i]     == h_aware[i]) xor_match_aware++;
    }
    printf("\n=== BIT-EXACT vs baseline (natural cp.async + LDS) ===\n");
    printf("TMA-swizzle+LDS (naive addr):                             %3d/128\n", lds_match);
    printf("TMA-swizzle+ldmatrix (naive addr):                        %3d/128\n", lm_match_base);
    printf("TMA-swizzle+LDS-aware (XOR read):                         %3d/128\n", aware_match);
    printf("MANUAL XOR cp.async+LDS (XOR write + XOR read):           %3d/128 ← ШАГ A gate (a)\n", xor_match_base);
    printf("MANUAL XOR vs TMA-aware (byte-identical layouts?):        %3d/128 ← ШАГ A gate (c)\n", xor_match_aware);

    cudaFree(d_in); cudaFree(d_out_tma_lds); cudaFree(d_out_tma_lm); cudaFree(d_out_base); cudaFree(d_out_aware); cudaFree(d_out_xor);
    return 0;
}
