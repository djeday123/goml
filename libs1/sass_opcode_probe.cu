/*
 * sass_opcode_probe.cu — Тест SM декодера на распознавание Hopper опкодов
 *
 * Идея: если RTL декодера шарится между sm_89 и sm_90,
 * то TMA/cluster опкоды будут РАСПОЗНАНЫ декодером (но не выполнены).
 *
 * Различаем:
 *   1. "Illegal Instruction" (trap 0x06) → декодер НЕ знает опкод
 *   2. Другая ошибка (trap 0x01/0x0E) → декодер знает, но execution unit нет
 *   3. Нет ошибки → hardware поддерживает!
 *
 * Метод:
 *   - Crafted inline PTX/SASS с разными кодировками
 *   - Мониторинг CUDA error type после запуска
 *   - Сравнение с заведомо мусорным опкодом (baseline)
 *
 * nvcc -arch=sm_89 -o sass_opcode_probe sass_opcode_probe.cu -lcuda
 * ./sass_opcode_probe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    do                                                                 \
    {                                                                  \
        cudaError_t e = (call);                                        \
        if (e != cudaSuccess)                                          \
        {                                                              \
            printf("  CUDA error %d: %s\n", e, cudaGetErrorString(e)); \
        }                                                              \
    } while (0)

#define CHECK_DRV(call)                              \
    do                                               \
    {                                                \
        CUresult e = (call);                         \
        if (e != CUDA_SUCCESS)                       \
        {                                            \
            const char *s;                           \
            cuGetErrorString(e, &s);                 \
            printf("  Driver error %d: %s\n", e, s); \
        }                                            \
    } while (0)

// ============================================================================
// Test 1: Baseline — известные рабочие инструкции sm_89
// ============================================================================

__global__ void baseline_kernel(int *out)
{
    out[threadIdx.x] = threadIdx.x + 1;
}

// ============================================================================
// Test 2: cp.async (sm_80+, должно работать на sm_89)
// ============================================================================

__global__ void cp_async_kernel(int *out, const int *in)
{
    __shared__ int smem[256];
    int tid = threadIdx.x;

    // cp.async.ca.shared.global — supported sm_80+
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        "cp.async.commit_group;\n"
        "cp.async.wait_group 0;\n" ::"r"((unsigned)((char *)smem + tid * 4 - (char *)smem + (unsigned long long)smem)),
        "l"(in + tid));
    __syncthreads();
    out[tid] = smem[tid];
}

// ============================================================================
// Test 3: Попытка использовать mbarrier (Hopper cluster barrier)
//
// mbarrier.init — инициализация аппаратного барьера
// На Hopper это часть TMA pipeline
// Вопрос: примет ли ptxas для sm_89?
// ============================================================================

// PTX модули для загрузки через Driver API
// Так мы обходим nvcc ограничения на target

// 3a: mbarrier.init для sm_89 (не должен компилироваться, но пробуем)
const char *ptx_mbarrier_89 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_mbarrier(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 8 .b8 mbar[8];\n"
    "\n"
    "    // mbarrier.init — Hopper feature\n"
    "    // Если декодер знает: компилируется\n"
    "    // Если нет: ptxas error\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// 3b: mbarrier для sm_90 (должен компилироваться)
const char *ptx_mbarrier_90 =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_mbarrier(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 8 .b8 mbar[8];\n"
    "\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 4: cp.async.bulk — TMA bulk copy (Hopper only)
// ============================================================================

const char *ptx_bulk_copy_89 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_bulk(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 128 .b8 smem[256];\n"
    "    .shared .align 8 .b8 mbar[8];\n"
    "\n"
    "    // cp.async.bulk — Hopper TMA\n"
    "    // Это самый прямой TMA тест\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    cp.async.bulk.shared.global [smem], [%rd0], 256, [mbar];\n"
    "\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_bulk_copy_90 =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_bulk(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 128 .b8 smem[256];\n"
    "    .shared .align 8 .b8 mbar[8];\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    cp.async.bulk.shared.global [smem], [%rd0], 256, [mbar];\n"
    "\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 5: Cluster-specific PTX
// ============================================================================

const char *ptx_cluster_89 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_cluster(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<8>;\n"
    "\n"
    "    // Cluster dimension query — Hopper\n"
    "    mov.u32 %r0, %cluster_ctaid.x;\n"
    "    mov.u32 %r1, %cluster_nctaid.x;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

const char *ptx_cluster_90 =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_cluster(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<8>;\n"
    "\n"
    "    mov.u32 %r0, %cluster_ctaid.x;\n"
    "    mov.u32 %r1, %cluster_nctaid.x;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 6: setmaxnreg — dynamic register allocation (Hopper)
// ============================================================================

const char *ptx_setmaxnreg_89 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_setmaxnreg(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "\n"
    "    // setmaxnreg — Hopper dynamic register reallocation\n"
    "    setmaxnreg.inc.sync.aligned.u32 64;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 7: fence.proxy.tensormap — TMA descriptor fence (Hopper)
// ============================================================================

const char *ptx_tensormap_89 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_tensormap(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "\n"
    "    // fence.proxy.tensormap — Hopper TMA descriptor\n"
    "    fence.proxy.tensormap::generic.release.gpu;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// PTX module loader with detailed error reporting
// ============================================================================

typedef struct
{
    const char *name;
    const char *ptx;
    const char *target;
    const char *func_name;
    int hopper_only; // 1 = Hopper-only instruction
} PTXTest;

void test_ptx_module(PTXTest *test, int *d_out)
{
    printf("\n  ─── %s [%s] ───\n", test->name, test->target);

    CUmodule mod = NULL;
    CUfunction func = NULL;

    // JIT compile options
    char error_log[4096] = {0};
    char info_log[4096] = {0};

    CUjit_option opts[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_GENERATE_LINE_INFO,
    };
    void *vals[] = {
        error_log,
        (void *)(size_t)sizeof(error_log),
        info_log,
        (void *)(size_t)sizeof(info_log),
        (void *)(size_t)1,
    };

    CUresult res = cuModuleLoadDataEx(&mod, test->ptx, 5, opts, vals);

    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    JIT compile: FAILED — %s (CUresult=%d)\n", errStr, res);
        if (strlen(error_log) > 0)
        {
            printf("    JIT error log: %.200s\n", error_log);
        }

        // Ключевой момент: ПОЧЕМУ не скомпилировалось?
        if (res == CUDA_ERROR_INVALID_PTX)
        {
            printf("    → ptxas НЕ ЗНАЕТ эту инструкцию для %s\n", test->target);
            printf("    → Это означает: ptxas отказывается компилировать\n");
        }
        else if (res == CUDA_ERROR_NO_BINARY_FOR_GPU)
        {
            printf("    → Binary не совместим с текущим GPU\n");
        }
        else if (res == CUDA_ERROR_INVALID_IMAGE)
        {
            printf("    → Invalid image/format\n");
        }
        return;
    }

    printf("    JIT compile: SUCCESS ✓\n");
    if (strlen(info_log) > 0)
    {
        printf("    JIT info: %.200s\n", info_log);
    }

    // Попробуем получить функцию
    res = cuModuleGetFunction(&func, mod, test->func_name);
    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    GetFunction: FAILED — %s\n", errStr);
        cuModuleUnload(mod);
        return;
    }
    printf("    GetFunction: SUCCESS ✓\n");

    // Попробуем ЗАПУСТИТЬ
    // Это ключевой момент — JIT мог скомпилировать, но hardware может отвергнуть
    void *args[] = {&d_out};

    printf("    Launching kernel...\n");
    res = cuLaunchKernel(func, 1, 1, 1, // grid
                         1, 1, 1,       // block (1 thread)
                         0, 0,          // shared mem, stream
                         args, NULL);

    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    Launch: FAILED — %s (CUresult=%d)\n", errStr, res);
        printf("    → Driver rejected launch BEFORE hardware execution\n");
        cuModuleUnload(mod);
        return;
    }
    printf("    Launch: submitted ✓\n");

    // Sync and check for hardware errors
    res = cuCtxSynchronize();
    if (res != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("    Sync: HARDWARE ERROR — %s (CUresult=%d)\n", errStr, res);

        // Decode the error
        if (res == CUDA_ERROR_ILLEGAL_INSTRUCTION)
        {
            printf("    >>> ILLEGAL_INSTRUCTION — декодер НЕ распознал опкод\n");
            printf("    >>> SM_89 decoder does NOT have this opcode\n");
        }
        else if (res == CUDA_ERROR_ILLEGAL_ADDRESS)
        {
            printf("    >>> ILLEGAL_ADDRESS — декодер РАСПОЗНАЛ опкод,\n");
            printf("    >>> но execution path обратился к несуществующему ресурсу!\n");
            printf("    >>> ЭТО ЗНАЧИТ ЛОГИКА В ДЕКОДЕРЕ ЕСТЬ!\n");
        }
        else if (res == CUDA_ERROR_HARDWARE_STACK_ERROR)
        {
            printf("    >>> HARDWARE_STACK_ERROR — execution unit есть но сбой\n");
        }
        else if (res == CUDA_ERROR_LAUNCH_FAILED)
        {
            printf("    >>> LAUNCH_FAILED — general execution failure\n");
            printf("    >>> Нужно выяснить конкретный тип ошибки\n");
        }
        else
        {
            printf("    >>> Error code %d — нестандартная ошибка\n", res);
        }
    }
    else
    {
        // Успех!
        int result;
        cuMemcpyDtoH(&result, (CUdeviceptr)d_out, sizeof(int));
        printf("    Sync: SUCCESS ✓ — kernel executed! result=%d\n", result);

        if (test->hopper_only)
        {
            printf("    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            printf("    !!! HOPPER INSTRUCTION EXECUTED ON ADA !!!\n");
            printf("    !!! HIDDEN HARDWARE CONFIRMED !!!\n");
            printf("    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        }
    }

    cuModuleUnload(mod);
}

// ============================================================================
// Test 8: Brute-force SM capability register dump via CUDA
// ============================================================================

__global__ void read_sm_caps(uint32_t *out)
{
    // Прочитаем SM-specific special registers
    uint32_t val;

    // %smid — SM ID
    asm volatile("mov.u32 %0, %smid;" : "=r"(val));
    out[0] = val;

    // %nsmid — number of SMs
    asm volatile("mov.u32 %0, %nsmid;" : "=r"(val));
    out[1] = val;

    // %lanemask_eq/lt/le/ge/gt
    asm volatile("mov.u32 %0, %lanemask_eq;" : "=r"(val));
    out[2] = val;

    // %dynamic_smem_size
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(val));
    out[3] = val;

    // %total_smem_size
    asm volatile("mov.u32 %0, %total_smem_size;" : "=r"(val));
    out[4] = val;

    // %aggr_smem_size (if available)
    // This is Hopper-only — aggregate shared memory across cluster
    // asm volatile("mov.u32 %0, %aggr_smem_size;" : "=r"(val));
    // out[5] = val;
}

// ============================================================================
// Test 9: Check exact trap code via exception handler
// ============================================================================

const char *ptx_garbage_opcode =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_garbage(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    mov.u32 %r0, 42;\n"
    "    // Попробуем trap instruction для baseline\n"
    "    trap;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 10: Check mbarrier.init via sm_89 — this instruction
// actually exists on sm_80+ in limited form!
// ============================================================================

const char *ptx_mbarrier_init_80 =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_mbarrier_init(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "\n"
    "    // На sm_80+ mbarrier.init МОЖЕТ быть доступен\n"
    "    // (он часть async copy pipeline, не только кластеров)\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 11: mbarrier.arrive (the async completion signal that TMA uses)
// ============================================================================

const char *ptx_mbarrier_arrive =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_mbarrier_arrive(\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 8 .b64 mbar;\n"
    "\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "    mbarrier.arrive.shared.b64 %rd1, [mbar];\n"
    "\n"
    "    ld.param.u64 %rd0, [out_ptr];\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 12: Try cp.async with mbarrier (the TMA-light path)
// ============================================================================

const char *ptx_cpasync_mbarrier =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry test_cpasync_mbar(\n"
    "    .param .u64 src_ptr,\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 16 .b8 smem[64];\n"
    "    .shared .align 8 .b64 mbar;\n"
    "\n"
    "    ld.param.u64 %rd0, [src_ptr];\n"
    "    ld.param.u64 %rd1, [out_ptr];\n"
    "\n"
    "    mbarrier.init.shared.b64 [mbar], 1;\n"
    "\n"
    "    // cp.async с mbarrier completion — TMA-lite\n"
    "    cp.async.mbarrier.arrive.shared.b64 [mbar];\n"
    "\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd1], %r0;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Main
// ============================================================================

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  SASS Opcode Decoder Probe — Does sm_89 know sm_90 ops?   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Init CUDA
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS)
    {
        printf("cuInit failed: %d\n", res);
        return 1;
    }

    CUdevice dev;
    CUcontext ctx;
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    // Device info
    char name[256];
    int major, minor;
    cuDeviceGetName(name, sizeof(name), dev);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    printf("Device: %s (sm_%d%d)\n\n", name, major, minor);

    // Allocate device memory for results
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, 4096);
    cuMemsetD32(d_out, 0, 1024);
    int *d_out_ptr = (int *)(void *)d_out;

    // ============================================================
    // Phase 1: SM special registers
    // ============================================================
    printf("═══ Phase 1: SM Special Registers ═══\n\n");
    {
        uint32_t h_out[8] = {0};
        uint32_t *d_caps;
        cudaMalloc(&d_caps, sizeof(h_out));
        read_sm_caps<<<1, 1>>>(d_caps);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_caps, sizeof(h_out), cudaMemcpyDeviceToHost);

        printf("  %%smid           = %u\n", h_out[0]);
        printf("  %%nsmid          = %u (total SMs)\n", h_out[1]);
        printf("  %%lanemask_eq    = 0x%08X\n", h_out[2]);
        printf("  %%dynamic_smem   = %u\n", h_out[3]);
        printf("  %%total_smem     = %u\n", h_out[4]);
        cudaFree(d_caps);
    }

    // ============================================================
    // Phase 2: PTX compilation + execution tests
    // ============================================================
    printf("\n═══ Phase 2: PTX Opcode Tests ═══\n");
    printf("  Testing which Hopper PTX instructions compile/execute on sm_89\n");

    PTXTest tests[] = {
        // Baseline: trap instruction (should compile, should trap)
        {"trap (baseline)", ptx_garbage_opcode, "sm_89", "test_garbage", 0},

        // mbarrier.init — exists on sm_80+ in limited form?
        {"mbarrier.init [sm_89]", ptx_mbarrier_init_80, "sm_89", "test_mbarrier_init", 1},

        // mbarrier with arrive
        {"mbarrier.init+arrive [sm_89]", ptx_mbarrier_arrive, "sm_89", "test_mbarrier_arrive", 1},

        // mbarrier for sm_89 target
        {"mbarrier.init [sm_89 target]", ptx_mbarrier_89, "sm_89", "test_mbarrier", 1},

        // mbarrier for sm_90 target (cross-compile)
        {"mbarrier.init [sm_90 target]", ptx_mbarrier_90, "sm_90", "test_mbarrier", 1},

        // cp.async.bulk for sm_89
        {"cp.async.bulk [sm_89]", ptx_bulk_copy_89, "sm_89", "test_bulk", 1},

        // cp.async.bulk for sm_90
        {"cp.async.bulk [sm_90]", ptx_bulk_copy_90, "sm_90", "test_bulk", 1},

        // cluster ID registers for sm_89
        {"cluster_ctaid [sm_89]", ptx_cluster_89, "sm_89", "test_cluster", 1},

        // cluster ID registers for sm_90
        {"cluster_ctaid [sm_90]", ptx_cluster_90, "sm_90", "test_cluster", 1},

        // setmaxnreg for sm_89
        {"setmaxnreg [sm_89]", ptx_setmaxnreg_89, "sm_89", "test_setmaxnreg", 1},

        // fence.proxy.tensormap for sm_89
        {"fence.proxy.tensormap [sm_89]", ptx_tensormap_89, "sm_89", "test_tensormap", 1},

        // cp.async + mbarrier combo
        {"cp.async.mbarrier [sm_89]", ptx_cpasync_mbarrier, "sm_89", "test_cpasync_mbar", 1},

        {NULL, NULL, NULL, NULL, 0}};

    for (int i = 0; tests[i].name; i++)
    {
        test_ptx_module(&tests[i], d_out_ptr);

        // Reset context after potential hardware error
        cuCtxSynchronize();
    }

    // ============================================================
    // Phase 3: Full attribute scan for cluster/TMA capabilities
    // ============================================================
    printf("\n\n═══ Phase 3: Cluster/TMA-related Device Attributes ═══\n\n");

    struct
    {
        int id;
        const char *name;
    } cluster_attrs[] = {
        {95, "CLUSTER_LAUNCH (CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH)"},
        {96, "DEFERRED_MAPPING"},
        {97, "MAX_SMEM_PER_BLOCK_OPTIN"},
        {106, "MAX_BLOCKS_PER_MULTIPROCESSOR"},
        {111, "RESERVED_SHARED_MEMORY_PER_BLOCK"},
        {112, "SPARSE_CUDA_ARRAY_SUPPORTED"},
        {113, "READ_ONLY_HOST_REGISTER_SUPPORTED"},
        {114, "TIMELINE_SEMAPHORE_INTEROP_SUPPORTED"},
        {115, "MEMORY_POOLS_SUPPORTED"},
        {117, "GPU_DIRECT_RDMA_SUPPORTED"},
        {-1, NULL}};

    for (int i = 0; cluster_attrs[i].name; i++)
    {
        int val;
        CUresult r = cuDeviceGetAttribute(&val, (CUdevice_attribute)cluster_attrs[i].id, dev);
        if (r == CUDA_SUCCESS)
        {
            printf("  attr[%3d] %-50s = %d",
                   cluster_attrs[i].id, cluster_attrs[i].name, val);
            if (cluster_attrs[i].id == 95 && val == 1)
                printf("  <<< CLUSTER LAUNCH SUPPORTED!");
            printf("\n");
        }
    }

    // ============================================================
    // Summary
    // ============================================================
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  INTERPRETATION                                            ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                            ║\n");
    printf("║  Три уровня проверки:                                      ║\n");
    printf("║                                                            ║\n");
    printf("║  1. PTX → SASS compile (ptxas)                             ║\n");
    printf("║     FAILED = ptxas не знает инструкцию для target sm       ║\n");
    printf("║     SUCCESS = ptxas генерирует SASS код                    ║\n");
    printf("║                                                            ║\n");
    printf("║  2. Kernel launch                                          ║\n");
    printf("║     FAILED = driver отказывает (sm mismatch)               ║\n");
    printf("║     SUCCESS = driver пропустил на hardware                  ║\n");
    printf("║                                                            ║\n");
    printf("║  3. Hardware execution                                     ║\n");
    printf("║     ILLEGAL_INSTRUCTION = decoder НЕ знает опкод           ║\n");
    printf("║     ILLEGAL_ADDRESS = decoder знает, unit НЕТ              ║\n");
    printf("║     LAUNCH_FAILED = execution error (investigate!)         ║\n");
    printf("║     SUCCESS = hardware поддерживает!                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    cuMemFree(d_out);
    cuCtxDestroy(ctx);

    return 0;
}
