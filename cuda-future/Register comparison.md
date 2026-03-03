# FA2 vs v54 Register & Resource Comparison

## cuobjdump Analysis (March 2026)

### FA2 Forward Kernel (hdim=128, half_t, sm_80)
```
flash_fwd_kernel<128,128,64,4,half_t> variants:
  Lb0ELb1ELb0ELb0ELb0ELb0ELb0ELb0E: REG:255 STACK:200 SHARED:1024 (sm_90)
  Lb0ELb1ELb0ELb0ELb0ELb0ELb0ELb0E: REG:255 STACK:264 SHARED:0   (sm_80)
  Lb0ELb1ELb0ELb0ELb0ELb0ELb1ELb0E: REG:255 STACK:280 SHARED:0   (sm_80)
  Lb0ELb1ELb0ELb1ELb0ELb0ELb0ELb0E: REG:255 STACK:152 SHARED:0   (sm_80)
```

### v54 (our production kernel)
```
fa54_kernel: REG:190 STACK:0 SHARED:32768
fa20_kernel: REG:192 STACK:0 SHARED:32768
```

### v47 (occupancy=3 experiment)  
```
fa47_kernel: REG:168 STACK:0 SHARED:32768
```

### v20 (baseline)
```
fa20_kernel: REG:192 STACK:0 SHARED:32768
```

## Occupancy Analysis (SM89: 65536 regs, 100KB smem)

| Kernel | Regs | Regs/Block | Blocks/SM | SMEM/Block | Spills |
|--------|------|------------|-----------|------------|--------|
| FA2    | 255  | 32640      | 1*        | ~1KB       | YES    |
| v54    | 190  | 24320      | 2         | 32KB       | NO     |
| v47    | 168  | 21504      | 3         | 32KB       | NO     |
| v20    | 192  | 24576      | 2         | 32KB       | NO     |

* FA2 at 255 regs: 32640 regs × 2 = 65280 < 65536, so technically 2 blocks fit
  BUT stack spills add local memory pressure, effective occupancy likely 1-2

## Key Insight
FA2 trades occupancy for register count (more live variables in registers)
and compensates with CUTLASS pipeline scheduling.

Our v54 has better occupancy but nvcc's auto-scheduling can't match
CUTLASS's explicit instruction interleaving.

## Performance
- FA2: 159T (96.4% of 165T peak)
- v54: 153T (92.5% of 165T peak)  
- Gap: 6T (3.8%)

## Note on SM89 Compatibility
FA2 .so contains NO sm_89 fatbin — only sm_80 and sm_90.
On RTX 4090 (sm_89), FA2 runs sm_80 code via forward compatibility.
This means FA2 doesn't use SM89-specific features (like FP8 tensor cores).
