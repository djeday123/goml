#!/bin/bash
# cubin_patch_test.sh
# Compile PTX for sm_90 with ptxas, patch cubin to sm_89, test on RTX 4090
#
# This bypasses ptxas sm_89 restrictions by:
# 1. Compiling PTX -> cubin for sm_90 (ptxas allows all instructions)
# 2. Binary patching the SM version in ELF header to sm_89
# 3. Loading patched cubin via cuModuleLoadData on RTX 4090

set -e

WORKDIR="/tmp/cubin_patch"
mkdir -p "$WORKDIR"

echo "==========================================================="
echo "  Binary Patch Test - Bypass ptxas sm_89 restrictions"
echo "==========================================================="

# ============================================================
# Step 1: Write PTX files for sm_90 target
# ============================================================

# Test A: mbarrier.try_wait.parity (blocked on sm_89)
cat > "$WORKDIR/test_try_wait.ptx" << 'EOF'
.version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<4>;
    .reg .b32 %r<4>;
    .reg .pred %p<2>;
    .shared .align 8 .b64 mbar;
    mbarrier.init.shared.b64 [mbar], 1;
    mbarrier.arrive.shared.b64 %rd1, [mbar];
WAIT:
    mbarrier.try_wait.parity.shared.b64 %p0, [mbar], 0;
    @!%p0 bra WAIT;
    ld.param.u64 %rd0, [p];
    mov.u32 %r0, 100;
    st.global.u32 [%rd0], %r0;
    ret;
}
EOF

# Test B: mbarrier.arrive.expect_tx (TMA transaction count)
cat > "$WORKDIR/test_expect_tx.ptx" << 'EOF'
.version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<4>;
    .reg .b32 %r<2>;
    .shared .align 8 .b64 mbar;
    mbarrier.init.shared.b64 [mbar], 1;
    mbarrier.arrive.expect_tx.shared.b64 %rd1, [mbar], 256;
    mbarrier.arrive.shared.b64 %rd1, [mbar];
    ld.param.u64 %rd0, [p];
    mov.u32 %r0, 101;
    st.global.u32 [%rd0], %r0;
    ret;
}
EOF

# Test C: cluster_ctaid (cluster register)
cat > "$WORKDIR/test_cluster.ptx" << 'EOF'
.version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<2>;
    .reg .b32 %r<4>;
    mov.u32 %r0, %cluster_ctaid.x;
    mov.u32 %r1, %cluster_nctaid.x;
    ld.param.u64 %rd0, [p];
    st.global.u32 [%rd0], %r0;
    ret;
}
EOF

# Test D: setmaxnreg
cat > "$WORKDIR/test_setmaxnreg.ptx" << 'EOF'
.version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<2>;
    .reg .b32 %r<2>;
    setmaxnreg.inc.sync.aligned.u32 64;
    ld.param.u64 %rd0, [p];
    mov.u32 %r0, 102;
    st.global.u32 [%rd0], %r0;
    ret;
}
EOF

# Test E: ld.shared::cluster (DSMEM)
cat > "$WORKDIR/test_dsmem.ptx" << 'EOF'
.version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<2>;
    .reg .b32 %r<4>;
    .shared .align 4 .b32 sdata;
    mov.u32 %r1, 42;
    st.shared.u32 [sdata], %r1;
    bar.sync 0;
    ld.shared::cluster.u32 %r0, [sdata];
    ld.param.u64 %rd0, [p];
    st.global.u32 [%rd0], %r0;
    ret;
}
EOF

# Test F: simple sm_90 baseline (no hopper-only instructions)
cat > "$WORKDIR/test_baseline90.ptx" << 'EOF'
.version 8.0
.target sm_90
.address_size 64
.visible .entry k(.param .u64 p) {
    .reg .b64 %rd<2>;
    .reg .b32 %r<2>;
    ld.param.u64 %rd0, [p];
    mov.u32 %r0, 99;
    st.global.u32 [%rd0], %r0;
    ret;
}
EOF

# ============================================================
# Step 2: Compile PTX -> cubin with ptxas for sm_90
# ============================================================

echo ""
echo "=== Step 2: Compiling PTX -> cubin for sm_90 ==="
echo ""

PTXAS=$(which ptxas 2>/dev/null || echo "/usr/local/cuda/bin/ptxas")

for test in baseline90 try_wait expect_tx cluster setmaxnreg dsmem; do
    PTX="$WORKDIR/test_${test}.ptx"
    CUBIN="$WORKDIR/test_${test}.cubin"
    
    echo -n "  Compiling $test... "
    if $PTXAS -arch=sm_90 -o "$CUBIN" "$PTX" 2>"$WORKDIR/${test}_err.txt"; then
        SIZE=$(stat -c%s "$CUBIN" 2>/dev/null || echo "?")
        echo "OK ($SIZE bytes)"
    else
        echo "FAILED"
        cat "$WORKDIR/${test}_err.txt" | head -3
    fi
done

# ============================================================
# Step 3: Analyze cubin ELF structure
# ============================================================

echo ""
echo "=== Step 3: Analyzing cubin ELF ==="
echo ""

# Check if baseline compiled
if [ ! -f "$WORKDIR/test_baseline90.cubin" ]; then
    echo "No cubin produced. ptxas may not support sm_90."
    echo "Trying with nvcc..."
    
    # Alternative: use nvcc to produce cubin
    cat > "$WORKDIR/baseline90.cu" << 'CUDA_EOF'
extern "C" __global__ void k(unsigned long long *p) {
    *p = 99;
}
CUDA_EOF
    
    nvcc -arch=sm_90 -cubin -o "$WORKDIR/test_baseline90.cubin" "$WORKDIR/baseline90.cu" 2>&1 || echo "nvcc sm_90 also failed"
fi

if [ -f "$WORKDIR/test_baseline90.cubin" ]; then
    echo "  baseline90.cubin ELF header:"
    xxd -l 128 "$WORKDIR/test_baseline90.cubin"
    echo ""
    
    # Find SM version in ELF
    echo "  Looking for SM version markers..."
    
    # In NVIDIA cubin ELF:
    # - e_flags field (offset 0x30 in ELF64) contains SM version
    # - Also in .nv.info section
    
    # Read e_flags (ELF64: offset 0x30, 4 bytes)
    E_FLAGS=$(xxd -s 0x30 -l 4 -p "$WORKDIR/test_baseline90.cubin")
    echo "  e_flags = 0x$E_FLAGS"
    
    # Read ELF header fields
    echo "  ELF class/data/version:"
    xxd -s 4 -l 4 "$WORKDIR/test_baseline90.cubin"
    
    # Dump first sections looking for SM version
    echo ""
    echo "  Searching for 0x5A (sm_90) in cubin..."
    grep -boa $'\x5a' "$WORKDIR/test_baseline90.cubin" | head -20
    
    echo ""
    echo "  readelf sections:"
    readelf -S "$WORKDIR/test_baseline90.cubin" 2>/dev/null | head -30
    
    echo ""
    echo "  readelf header:"
    readelf -h "$WORKDIR/test_baseline90.cubin" 2>/dev/null
    
    echo ""
    echo "  cuobjdump info:"
    cuobjdump "$WORKDIR/test_baseline90.cubin" 2>/dev/null | head -20
fi

# ============================================================
# Step 4: Patch cubin SM version and load
# ============================================================

echo ""
echo "=== Step 4: Building loader ==="
echo ""

cat > "$WORKDIR/patch_loader.cu" << 'LOADER_EOF'
/*
 * patch_loader.cu - Load cubin files, optionally patch SM version
 *
 * Reads a cubin file, patches e_flags SM version, loads via cuModuleLoadData
 *
 * Usage: ./patch_loader <cubin_file> [patch_sm]
 *   patch_sm: target SM version to patch to (e.g. 89)
 *             0 = don't patch, try loading as-is
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

// ELF64 header offsets
#define ELF_MAGIC_OFFSET    0
#define ELF_CLASS_OFFSET    4
#define ELF_FLAGS_OFFSET    0x30  // e_flags in ELF64

// NVIDIA cubin e_flags encoding:
// Bits [7:0]   = SM minor version
// Bits [15:8]  = SM major version  
// (This may vary - we'll probe)

unsigned char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return NULL; }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(*size);
    fread(buf, 1, *size, f);
    fclose(f);
    return buf;
}

void hexdump(const unsigned char *buf, size_t off, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (i % 16 == 0) printf("    %04zx: ", off + i);
        printf("%02x ", buf[off + i]);
        if (i % 16 == 15 || i == len-1) printf("\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <cubin_file> [patch_sm_version]\n", argv[0]);
        printf("  patch_sm_version: e.g. 89 to patch SM to 8.9\n");
        printf("  0 or omit = load without patching\n");
        return 1;
    }
    
    const char *cubin_path = argv[1];
    int patch_sm = (argc > 2) ? atoi(argv[2]) : 0;
    
    printf("=== Loading cubin: %s ===\n", cubin_path);
    if (patch_sm) printf("=== Will patch SM to %d.%d ===\n", patch_sm/10, patch_sm%10);
    
    // Read cubin
    size_t size;
    unsigned char *cubin = read_file(cubin_path, &size);
    if (!cubin) return 1;
    printf("  Size: %zu bytes\n", size);
    
    // Verify ELF
    if (cubin[0] != 0x7f || cubin[1] != 'E' || cubin[2] != 'L' || cubin[3] != 'F') {
        printf("  ERROR: Not an ELF file!\n");
        free(cubin);
        return 1;
    }
    printf("  Valid ELF magic\n");
    
    // Read e_flags
    unsigned int e_flags = *(unsigned int *)(cubin + ELF_FLAGS_OFFSET);
    printf("  Original e_flags = 0x%08X\n", e_flags);
    
    // Decode SM version from e_flags
    // NVIDIA encoding: e_flags contains SM version
    // Common pattern: lower byte = SM version (0x5A = 90, 0x59 = 89)
    // But it could be different...
    
    printf("  e_flags byte breakdown:\n");
    printf("    [0x30] = 0x%02X\n", cubin[ELF_FLAGS_OFFSET]);
    printf("    [0x31] = 0x%02X\n", cubin[ELF_FLAGS_OFFSET+1]);
    printf("    [0x32] = 0x%02X\n", cubin[ELF_FLAGS_OFFSET+2]);
    printf("    [0x33] = 0x%02X\n", cubin[ELF_FLAGS_OFFSET+3]);
    
    // Show region around e_flags for analysis
    printf("\n  ELF header (0x28-0x3F):\n");
    hexdump(cubin, 0x28, 24);
    
    // Search for SM version markers in entire file
    printf("\n  Searching for SM90 markers (0x5A=90, 0x59=89)...\n");
    int count_5a = 0, count_59 = 0;
    for (size_t i = 0; i < size; i++) {
        if (cubin[i] == 0x5A && i < 256) {
            printf("    0x5A at offset 0x%04zx", i);
            if (i > 0) printf(" (prev=0x%02x)", cubin[i-1]);
            if (i+1 < size) printf(" (next=0x%02x)", cubin[i+1]);
            printf("\n");
            count_5a++;
        }
    }
    printf("    Total 0x5A in first 256 bytes: %d\n", count_5a);
    
    // Also look for SM version in NVIDIA-specific sections
    // .nv.info contains SM version in a structured format
    printf("\n  Searching for 'sm_90' / 'sm_89' strings...\n");
    for (size_t i = 0; i < size - 5; i++) {
        if (memcmp(cubin + i, "sm_90", 5) == 0) {
            printf("    'sm_90' at offset 0x%04zx\n", i);
        }
        if (memcmp(cubin + i, "sm_89", 5) == 0) {
            printf("    'sm_89' at offset 0x%04zx\n", i);
        }
    }
    
    // ============================================================
    // Patch if requested
    // ============================================================
    
    if (patch_sm) {
        printf("\n=== Patching SM version ===\n");
        
        // Strategy: find all occurrences of SM 90 encoding and replace with 89
        // The exact encoding depends on NVIDIA's format
        
        // Method 1: Patch e_flags
        // Try: e_flags lower byte might be SM*10 or SM major
        unsigned int new_flags = e_flags;
        
        // If e_flags contains 0x5A (90 decimal) somewhere, replace with 0x59 (89)
        unsigned char *fp = (unsigned char *)&new_flags;
        int patched_eflags = 0;
        for (int i = 0; i < 4; i++) {
            if (fp[i] == 0x5A) { // 90
                fp[i] = 0x59;    // 89
                patched_eflags = 1;
            }
        }
        
        if (patched_eflags) {
            printf("  Patched e_flags: 0x%08X -> 0x%08X\n", e_flags, new_flags);
            *(unsigned int *)(cubin + ELF_FLAGS_OFFSET) = new_flags;
        } else {
            printf("  No 0x5A found in e_flags, trying broader patch...\n");
            // Try different encoding: maybe SM version = e_flags & 0xFF
            printf("  e_flags & 0xFF = %d\n", e_flags & 0xFF);
            printf("  e_flags & 0xFFFF = %d\n", e_flags & 0xFFFF);
            
            // Brute force: if lower bits = 90, change to 89
            if ((e_flags & 0xFF) == 90) {
                new_flags = (e_flags & ~0xFF) | 89;
                printf("  Patched e_flags (low byte): 0x%08X -> 0x%08X\n", e_flags, new_flags);
                *(unsigned int *)(cubin + ELF_FLAGS_OFFSET) = new_flags;
            } else if ((e_flags & 0xFFFF) == 90) {
                new_flags = (e_flags & ~0xFFFF) | 89;
                printf("  Patched e_flags (low word): 0x%08X -> 0x%08X\n", e_flags, new_flags);
                *(unsigned int *)(cubin + ELF_FLAGS_OFFSET) = new_flags;
            }
        }
        
        // Method 2: Patch ALL occurrences of SM version in the binary
        // Look for the 4-byte patterns that encode SM 90
        int total_patches = 0;
        
        // Scan for .nv.info SM version entries
        // Format: 04 xx yy 00 where xx*10+yy = SM version
        // Or just 0x5A bytes near version-like context
        
        for (size_t i = 0; i < size; i++) {
            // Pattern: byte == 0x5A (90) preceded or followed by version-like context
            // Be conservative - only patch in ELF metadata areas
            if (cubin[i] == 0x5A) {
                // Check if this looks like an SM version field
                // Heuristic: near other small values, in first 1KB or in section headers
                if (i < 0x100 || 
                    (i > 0 && cubin[i-1] == 0x00 && (i+1 >= size || cubin[i+1] < 0x10))) {
                    printf("  Patching 0x5A -> 0x59 at offset 0x%04zx\n", i);
                    cubin[i] = 0x59;
                    total_patches++;
                }
            }
        }
        
        printf("  Total patches applied: %d\n", total_patches);
        
        // Save patched cubin
        char patched_path[512];
        snprintf(patched_path, sizeof(patched_path), "%s.patched", cubin_path);
        FILE *f = fopen(patched_path, "wb");
        fwrite(cubin, 1, size, f);
        fclose(f);
        printf("  Saved patched cubin to: %s\n", patched_path);
    }
    
    // ============================================================
    // Load cubin via CUDA Driver API
    // ============================================================
    
    printf("\n=== Loading cubin on GPU ===\n");
    
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        printf("  cuInit failed: %d\n", res);
        free(cubin);
        return 1;
    }
    
    CUdevice dev;
    CUcontext ctx;
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);
    
    char devname[256];
    cuDeviceGetName(devname, sizeof(devname), dev);
    printf("  Device: %s\n", devname);
    
    // Try loading
    CUmodule mod = NULL;
    res = cuModuleLoadData(&mod, cubin);
    
    if (res != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("  cuModuleLoadData: FAILED [%d] %s\n", res, errStr);
        
        if (res == CUDA_ERROR_INVALID_PTX) {
            printf("  ==> Driver rejected binary (PTX/cubin validation failed)\n");
        } else if (res == CUDA_ERROR_NO_BINARY_FOR_GPU) {
            printf("  ==> SM version mismatch - driver won't load\n");
            printf("  ==> This is the DRIVER GATE (level 2)\n");
        } else if (res == CUDA_ERROR_INVALID_IMAGE) {
            printf("  ==> Invalid binary image (corrupted by patching?)\n");
        }
        
        cuCtxDestroy(ctx);
        free(cubin);
        return 1;
    }
    
    printf("  cuModuleLoadData: SUCCESS!\n");
    
    // Get function
    CUfunction func;
    res = cuModuleGetFunction(&func, mod, "k");
    if (res != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("  cuModuleGetFunction: FAILED [%d] %s\n", res, errStr);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cubin);
        return 1;
    }
    printf("  cuModuleGetFunction: SUCCESS!\n");
    
    // Allocate and launch
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, 4096);
    cuMemsetD32(d_out, 0, 1024);
    
    void *args[] = { &d_out };
    
    printf("  Launching...\n");
    res = cuLaunchKernel(func, 1,1,1, 1,1,1, 1024, 0, args, NULL);
    if (res != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("  Launch: FAILED [%d] %s\n", res, errStr);
        cuModuleUnload(mod);
        cuMemFree(d_out);
        cuCtxDestroy(ctx);
        free(cubin);
        return 1;
    }
    
    res = cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("  Exec: ERROR [%d] %s\n", res, errStr);
        
        if (res == CUDA_ERROR_ILLEGAL_INSTRUCTION)
            printf("  ==> ILLEGAL_INSTRUCTION: SM decoder does NOT know this opcode!\n");
        else if (res == CUDA_ERROR_ILLEGAL_ADDRESS)
            printf("  ==> ILLEGAL_ADDRESS: decoder KNOWS opcode, execution unit issue\n");
        else if (res == CUDA_ERROR_LAUNCH_FAILED)
            printf("  ==> LAUNCH_FAILED: general hw execution error\n");
    } else {
        int result = 0;
        cuMemcpyDtoH(&result, d_out, sizeof(int));
        printf("  Exec: SUCCESS! result=%d\n", result);
        printf("  *** SM_90 INSTRUCTION EXECUTED ON SM_89 HARDWARE! ***\n");
    }
    
    cuModuleUnload(mod);
    cuMemFree(d_out);
    cuCtxDestroy(ctx);
    free(cubin);
    
    return 0;
}
LOADER_EOF

echo "  Compiling patch_loader..."
nvcc -arch=sm_89 -o "$WORKDIR/patch_loader" "$WORKDIR/patch_loader.cu" -lcuda
echo "  Done."

# ============================================================
# Step 5: Test loading
# ============================================================

echo ""
echo "=== Step 5: Testing cubin loading ==="
echo ""

# First: try loading sm_90 cubin as-is (expect: NO_BINARY_FOR_GPU)
for test in baseline90 try_wait expect_tx cluster setmaxnreg dsmem; do
    CUBIN="$WORKDIR/test_${test}.cubin"
    if [ -f "$CUBIN" ]; then
        echo "--- Testing $test (unpatched sm_90) ---"
        "$WORKDIR/patch_loader" "$CUBIN" 0
        echo ""
        
        echo "--- Testing $test (patched to sm_89) ---"
        "$WORKDIR/patch_loader" "$CUBIN" 89
        echo ""
    fi
done

echo "==========================================================="
echo "  DONE"
echo "==========================================================="