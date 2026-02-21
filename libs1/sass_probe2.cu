/*
 * sass_probe3.cu - Each test gets fresh CUDA context
 *
 * nvcc -arch=sm_89 -o sass_probe3 sass_probe3.cu -lcuda
 * ./sass_probe3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

static CUdevice g_dev;

typedef struct
{
    const char *name;
    const char *ptx;
    const char *func_name;
    int needs_src; // 1 = needs d_src arg
    int hopper_only;
} Test;

void run_one_test(Test *t)
{
    printf("  --- %s ---\n", t->name);

    // Fresh context for each test
    CUcontext ctx;
    CUresult res = cuCtxCreate(&ctx, 0, g_dev);
    if (res != CUDA_SUCCESS)
    {
        printf("    ctx create failed: %d\n", res);
        return;
    }

    CUdeviceptr d_out, d_src = 0;
    cuMemAlloc(&d_out, 4096);
    cuMemsetD32(d_out, 0, 1024);
    if (t->needs_src)
    {
        cuMemAlloc(&d_src, 4096);
        cuMemsetD32(d_src, 0xABCD, 1024);
    }

    // Compile
    char error_log[8192] = {0};
    CUjit_option opts[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void *vals[] = {
        error_log,
        (void *)(size_t)sizeof(error_log),
    };

    CUmodule mod = NULL;
    res = cuModuleLoadDataEx(&mod, t->ptx, 2, opts, vals);

    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    COMPILE: FAILED [%d] %s\n", res, errStr);
        if (strlen(error_log) > 0)
        {
            error_log[200] = 0;
            printf("    LOG: %s\n", error_log);
        }
        if (d_src)
            cuMemFree(d_src);
        cuMemFree(d_out);
        cuCtxDestroy(ctx);
        return;
    }
    printf("    COMPILE: OK\n");

    CUfunction func;
    res = cuModuleGetFunction(&func, mod, t->func_name);
    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    GETFUNC: FAILED [%d] %s\n", res, errStr);
        cuModuleUnload(mod);
        if (d_src)
            cuMemFree(d_src);
        cuMemFree(d_out);
        cuCtxDestroy(ctx);
        return;
    }

    // Launch
    void *args[2];
    if (t->needs_src)
    {
        args[0] = &d_src;
        args[1] = &d_out;
    }
    else
    {
        args[0] = &d_out;
    }

    res = cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 1024, 0, args, NULL);
    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    LAUNCH: FAILED [%d] %s\n", res, errStr);
        cuModuleUnload(mod);
        if (d_src)
            cuMemFree(d_src);
        cuMemFree(d_out);
        cuCtxDestroy(ctx);
        return;
    }

    // Sync
    res = cuCtxSynchronize();
    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    EXEC: ERROR [%d] %s\n", res, errStr);

        if (res == CUDA_ERROR_ILLEGAL_INSTRUCTION)
            printf("    ==> ILLEGAL_INSTRUCTION: decoder does NOT know opcode\n");
        else if (res == CUDA_ERROR_ILLEGAL_ADDRESS)
            printf("    ==> ILLEGAL_ADDRESS: decoder KNOWS opcode, unit missing\n");
        else if (res == CUDA_ERROR_LAUNCH_FAILED)
            printf("    ==> LAUNCH_FAILED: hw execution error\n");
        else if (res == CUDA_ERROR_MISALIGNED_ADDRESS)
            printf("    ==> MISALIGNED: decoder KNOWS opcode, address issue\n");
    }
    else
    {
        int result = 0;
        cuMemcpyDtoH(&result, d_out, sizeof(int));
        printf("    EXEC: SUCCESS! result=%d\n", result);
        if (t->hopper_only)
        {
            printf("    *** HOPPER FEATURE WORKS ON ADA sm_89! ***\n");
        }
    }

    cuModuleUnload(mod);
    if (d_src)
        cuMemFree(d_src);
    cuMemFree(d_out);
    cuCtxDestroy(ctx);
}

// ============================================================================
// PTX modules - all clean ASCII
// ============================================================================

const char *ptx_baseline =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<2>;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 1;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_trap =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    trap;\n"
    "    ret;\n"
    "}\n";

const char *ptx_mbar_init =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<2>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 2;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_mbar_arrive =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<2>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    mbarrier.arrive.shared.b64 %rd1, [mbar];\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 3;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_mbar_full =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .reg .pred %p<2>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    mbarrier.arrive.shared.b64 %rd1, [mbar];\n"
    "WAIT:\n"
    "    mbarrier.try_wait.parity.shared.b64 %p0, [mbar], 0;\n"
    "    @!%p0 bra WAIT;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 4;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_mbar_expect_tx =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<2>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    mbarrier.arrive.expect_tx.shared.b64 %rd1, [mbar], 256;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 5;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_mbar_nocomplete =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<2>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    mbarrier.init.shared.b64 [mbar], 2;\n"
    "    mbarrier.arrive.noComplete.shared.b64 %rd1, [mbar], 1;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 6;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_cpasync_mbar =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 src, .param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .reg .pred %p<2>;\n"
    "    .shared .align 16 .b8 smem[64];\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    ld.param.u64 %rd0, [src];\n"
    "    ld.param.u64 %rd1, [p];\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    cp.async.ca.shared.global [smem], [%rd0], 4;\n"
    "    cp.async.mbarrier.arrive.shared.b64 [mbar];\n"
    "    cp.async.commit_group;\n"
    "    cp.async.wait_group 0;\n"
    "WAIT:\n"
    "    mbarrier.try_wait.parity.shared.b64 %p0, [mbar], 0;\n"
    "    @!%p0 bra WAIT;\n"
    "    mov.u32 %r0, 7;\n"
    "    st.global.u32 [%rd1], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_bulk_89 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 src, .param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<2>;\n"
    "    .shared .align 128 .b8 smem[256];\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    ld.param.u64 %rd0, [src];\n"
    "    ld.param.u64 %rd1, [p];\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    cp.async.bulk.shared.global [smem], [%rd0], 256, [mbar];\n"
    "    mov.u32 %r0, 8;\n"
    "    st.global.u32 [%rd1], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_bulk_90 =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 src, .param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<2>;\n"
    "    .shared .align 128 .b8 smem[256];\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    ld.param.u64 %rd0, [src];\n"
    "    ld.param.u64 %rd1, [p];\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    cp.async.bulk.shared.global [smem], [%rd0], 256, [mbar];\n"
    "    mov.u32 %r0, 8;\n"
    "    st.global.u32 [%rd1], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_cluster_89 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<4>;\n"
    "    mov.u32 %r0, %cluster_ctaid.x;\n"
    "    mov.u32 %r1, %cluster_nctaid.x;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_cluster_90 =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<4>;\n"
    "    mov.u32 %r0, %cluster_ctaid.x;\n"
    "    mov.u32 %r1, %cluster_nctaid.x;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_setmaxnreg =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<2>;\n"
    "    setmaxnreg.inc.sync.aligned.u32 64;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 10;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_tensormap =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<2>;\n"
    "    fence.proxy.tensormap::generic.release.gpu;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 11;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_dsmem =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 4 .b32 sdata;\n"
    "    mov.u32 %r1, 42;\n"
    "    st.shared.u32 [sdata], %r1;\n"
    "    bar.sync 0;\n"
    "    ld.shared::cluster.u32 %r0, [sdata];\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// mbarrier.arrive.expect_tx.shared::cta (more specific qualifier)
const char *ptx_mbar_expect_tx_cta =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<2>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    mbarrier.arrive.expect_tx.shared.b64 %rd1, [mbar], 64;\n"
    "    mbarrier.arrive.shared.b64 %rd1, [mbar];\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 13;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

int main(void)
{
    printf("===========================================================\n");
    printf("  SASS Opcode Probe v3 - Fresh context per test\n");
    printf("  RTX 4090 (sm_89)\n");
    printf("===========================================================\n\n");

    cuInit(0);
    cuDeviceGet(&g_dev, 0);

    char name[256];
    cuDeviceGetName(name, sizeof(name), g_dev);
    printf("Device: %s\n\n", name);

    Test tests[] = {
        // Baselines
        {"baseline", ptx_baseline, "k", 0, 0},
        //{"trap skip", ptx_trap, "k", 0, 0},

        // mbarrier progressive
        {"mbarrier.init", ptx_mbar_init, "k", 0, 1},
        {"mbarrier.init+arrive", ptx_mbar_arrive, "k", 0, 1},
        {"mbarrier full cycle (init+arrive+try_wait)", ptx_mbar_full, "k", 0, 1},
        {"mbarrier.arrive.expect_tx (TMA tx count)", ptx_mbar_expect_tx, "k", 0, 1},
        {"mbarrier.arrive.expect_tx + arrive", ptx_mbar_expect_tx_cta, "k", 0, 1},
        {"mbarrier.arrive.noComplete", ptx_mbar_nocomplete, "k", 0, 1},

        // cp.async + mbarrier
        {"cp.async + mbarrier.arrive", ptx_cpasync_mbar, "k", 1, 1},

        // TMA bulk
        {"cp.async.bulk [sm_89]", ptx_bulk_89, "k", 1, 1},
        {"cp.async.bulk [sm_90]", ptx_bulk_90, "k", 1, 1},

        // Cluster
        {"cluster_ctaid [sm_89]", ptx_cluster_89, "k", 0, 1},
        {"cluster_ctaid [sm_90]", ptx_cluster_90, "k", 0, 1},

        // Hopper-only
        {"setmaxnreg", ptx_setmaxnreg, "k", 0, 1},
        {"fence.proxy.tensormap", ptx_tensormap, "k", 0, 1},
        {"ld.shared::cluster (DSMEM)", ptx_dsmem, "k", 0, 1},

        {NULL, NULL, NULL, 0, 0}};

    for (int i = 0; tests[i].name; i++)
    {
        run_one_test(&tests[i]);
        printf("\n");
    }

    // Cluster launch attribute check
    printf("=== Driver API Cluster Attributes ===\n");
    int val;
    cuDeviceGetAttribute(&val, (CUdevice_attribute)95, g_dev);
    printf("  CLUSTER_LAUNCH (attr 95) = %d\n", val);
    cuDeviceGetAttribute(&val, (CUdevice_attribute)96, g_dev);
    printf("  DEFERRED_MAPPING (attr 96) = %d\n", val);

    return 0;
}
