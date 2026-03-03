# CUDA Future Research

## Goal
Close the 6T gap: v54 (153T) → FA2 (159T) on RTX 4090 SM89.

## Root Cause
CUTLASS/CuTe instruction-level pipeline scheduling vs nvcc auto-scheduling.
FA2 uses explicit instruction interleaving; v54 relies on nvcc.

## FA2 Key Facts (from cuobjdump analysis)
- FA2 .so has NO sm_89 code — runs sm_80 compatibility on RTX 4090
- FA2 fwd hdim128 kernels: REG:255, STACK:152-320 bytes (spills!)
- FA2 runs at occupancy=1 with register spilling, still beats v54 occupancy=2
- This proves: scheduling quality > occupancy > register count

## v54 Key Facts
- REG:190, STACK:0, occupancy=2
- 153T = 92.5% peak, 96.2% of FA2
- No spills, good occupancy, but nvcc scheduling leaves 6T on table

## Approach
Compare SASS instruction patterns between FA2 and v54:
- FA2: expect interleaved LDGSTS/HMMA pattern (pipeline)
- v54: expect clustered loads then clustered computes (batched)

## SASS Key Instructions
```
HMMA.16816  = mma.sync (tensor core compute)
LDGSTS      = cp.async (global → shared async copy)  
LDG         = global memory load
LDS         = shared memory load
STS         = shared memory store
BAR.SYNC    = __syncthreads
```

## Files
- fa2_fwd_hdim128_sass.txt — FA2 forward kernel SASS dump (sm_80)
- v54_sass.txt — TODO: dump from server after compile
- register_comparison.md — cuobjdump register analysis

## Action Items
1. Dump v54 SASS on gpu4 server
2. Compare LDGSTS/HMMA interleaving patterns
3. Identify specific instruction reordering opportunities
4. (Optional) Write PTX kernel with manual scheduling
5. (Optional) Port to CUTLASS/CuTe primitives
