// Baseline kernel emitting QMMA.16816.F16.E4M3.E4M3 in a tight loop.
// We will compile this to a cubin, locate the QMMA SASS bytes, flip bit 76,
// and launch the patched cubin via the CUDA Driver API.
//
// Output buffer holds the final accumulator so we can verify "did the
// instruction actually compute?" — if patched bit 76 makes QMMA behave
// as m16n8k64 (taking 4× A bytes, 4× B bytes), the result will differ
// from the k=16 baseline given the same input pattern.

#include <cstdint>

extern "C" __global__ void qmma_baseline(
    const uint8_t * __restrict__ Aptr,
    const uint8_t * __restrict__ Bptr,
    uint16_t * __restrict__ Cptr,
    int iters)
{
    // Each warp loads one m16n8k16 tile of FP8 operands and one f16 accumulator.
    // A: 16x16 FP8 = 256 bytes = 8 .b32 per thread of 32-lane warp
    //    Actually per PTX mma m16n8k16 FP8: A frag = 8 elements per thread (2 .b32),
    //    B frag = 4 elements per thread (1 .b32), C frag = 2 .b16x2 (1 .b32 each).
    // But on Blackwell the canonical FP8 shape is m16n8k32: A=4.b32, B=2.b32.
    // We DELIBERATELY use the k16 shape because that's what nvdisasm calls
    // QMMA.16816 — the bit pattern we want to mutate.

    // Load operands as 32-bit registers via reinterpret cast on a per-thread basis.
    int lane = threadIdx.x & 31;
    const uint32_t *Au32 = reinterpret_cast<const uint32_t*>(Aptr) + lane * 2;
    const uint32_t *Bu32 = reinterpret_cast<const uint32_t*>(Bptr) + lane;
    uint32_t a0 = Au32[0], a1 = Au32[1];
    uint32_t b0 = Bu32[0];
    uint32_t c0 = 0, c1 = 0;

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        // m16n8k16, FP8 e4m3 × e4m3, FP16 accumulator.
        // This is QMMA.16816.F16.E4M3.E4M3 at SASS level — our target instruction.
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16 "
            "{%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(c0), "+r"(c1)
            : "r"(a0), "r"(a1), "r"(b0));
    }

    uint32_t *Cu32 = reinterpret_cast<uint32_t*>(Cptr) + lane * 2;
    Cu32[0] = c0;
    Cu32[1] = c1;
}
