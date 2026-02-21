/*
 * tma_ptx_test.cu - Попытка выполнить TMA инструкции на RTX 4090
 *
 * Стратегия:
 *   1. cp.async (без .bulk.tensor) — должен работать на sm_89
 *   2. cp.async.bulk — Hopper only, но попробуем
 *   3. Проверим через cuFuncGetAttribute скрытые capabilities
 *
 * nvcc -arch=sm_89 -o tma_ptx_test tma_ptx_test.cu
 * ./tma_ptx_test
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
    do                                                                       \
    {                                                                        \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess)                                                \
            printf("[%d] %s: %s\n", __LINE__, #call, cudaGetErrorString(e)); \
    } while (0)

#define CHECK_DRV(call)                                  \
    do                                                   \
    {                                                    \
        CUresult e = (call);                             \
        if (e != CUDA_SUCCESS)                           \
        {                                                \
            const char *s;                               \
            cuGetErrorString(e, &s);                     \
            printf("[%d] %s: %s\n", __LINE__, #call, s); \
        }                                                \
    } while (0)

// ============================================================================
// Test 1: cp.async — AsyncCopy (supported on sm_80+, including sm_89)
// Это НЕ TMA, но базовый async copy pipeline
// ============================================================================

__global__ void test_cp_async(float *dst, const float *src, int n)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n)
    {
        // cp.async — asynchronous copy from global to shared memory
        // Поддерживается на sm_80+ (Ampere, Ada, Hopper)
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((uint32_t)(uintptr_t)&smem[tid]),
            "l"(&src[gid]));

        // Commit and wait
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");

        __syncthreads();

        dst[gid] = smem[tid] * 2.0f;
    }
}

// ============================================================================
// Test 2: Попытка TMA-подобной bulk операции через inline PTX
// На sm_89 это должно вызвать illegal instruction если TMA нет
// ============================================================================

// НЕ КОМПИЛИРУЕТСЯ напрямую для sm_89, но мы проверяем через driver API
// Это PTX для sm_90 — мы попробуем загрузить через cuModuleLoadData

const char *tma_test_ptx =
    ".version 8.0\n"
    ".target sm_90\n" // Intentionally sm_90 — проверим загрузится ли
    ".address_size 64\n"
    "\n"
    ".visible .entry tma_kernel(.param .u64 desc_ptr, .param .u64 dst) {\n"
    "    .reg .b64 %rd<4>;\n"
    "    .reg .b32 %r<4>;\n"
    "    .shared .align 128 .b8 smem[1024];\n"
    "\n"
    "    ld.param.u64 %rd0, [desc_ptr];\n"
    "    ld.param.u64 %rd1, [dst];\n"
    "    mov.u32 %r0, 0;\n"
    "\n"
    "    // cp.async.bulk.tensor — это НАСТОЯЩИЙ TMA\n"
    "    // Если hardware поддерживает, выполнится\n"
    "    // Если нет — illegal instruction\n"
    "    // cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes\n"
    "    //     [smem], [desc_ptr, {%r0}], [mbar];\n"
    "\n"
    "    // Для безопасности просто проверим что sm_90 PTX загружается\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    ret;\n"
    "}\n";

// ============================================================================
// Test 3: Query hidden device attributes
// ============================================================================

void query_device_caps(void)
{
    printf("=== Device Capability Query ===\n\n");

    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("  Device: %s\n", prop.name);
    printf("  Compute: sm_%d%d\n", prop.major, prop.minor);
    printf("  Async Engine Count: %d\n", prop.asyncEngineCount);
    printf("  Concurrent Kernels: %d\n", prop.concurrentKernels);
    printf("  Cooperative Launch: %d\n", prop.cooperativeLaunch);
    printf("  Cluster Launch: %d\n", prop.clusterLaunch);
    printf("  Memory Pools Supported: %d\n", prop.memoryPoolsSupported);

    // Extended attributes via Driver API
    CUdevice cudev;
    CHECK_DRV(cuDeviceGet(&cudev, dev));

    printf("\n  Extended Attributes (Driver API):\n");

    // Все известные атрибуты, включая скрытые
    struct
    {
        CUdevice_attribute attr;
        const char *name;
    } attrs[] = {
        {CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, "ASYNC_ENGINE_COUNT"},
        {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "MAX_SMEM_PER_BLOCK_OPTIN"},
        {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "MAX_SMEM_PER_SM"},
        {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, "MAX_REGS_PER_BLOCK"},
        {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "MAX_REGS_PER_SM"},
        {CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, "MEMORY_POOLS"},
        {CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES, "MEMPOOL_HANDLE_TYPES"},
        // Hopper-specific attributes
        {(CUdevice_attribute)120, "CLUSTER_LAUNCH"},
        {(CUdevice_attribute)121, "DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED"},
        // Try high numbered attributes that might be TMA-related
        {(CUdevice_attribute)122, "ATTR_122 (unknown)"},
        {(CUdevice_attribute)123, "ATTR_123 (unknown)"},
        {(CUdevice_attribute)124, "ATTR_124 (unknown)"},
        {(CUdevice_attribute)125, "ATTR_125 (unknown)"},
        {(CUdevice_attribute)126, "ATTR_126 (unknown)"},
        {(CUdevice_attribute)127, "ATTR_127 (unknown)"},
        {(CUdevice_attribute)128, "ATTR_128 (unknown)"},
        {(CUdevice_attribute)129, "ATTR_129 (unknown)"},
        {(CUdevice_attribute)130, "ATTR_130 (unknown)"},
        {(CUdevice_attribute)131, "ATTR_131 (unknown)"},
        {(CUdevice_attribute)132, "ATTR_132 (TMA candidate?)"},
        {(CUdevice_attribute)133, "ATTR_133 (TMA candidate?)"},
        {(CUdevice_attribute)134, "ATTR_134 (TMA candidate?)"},
        {(CUdevice_attribute)135, "ATTR_135 (TMA candidate?)"},
        // Even higher — scan range
        {(CUdevice_attribute)140, "ATTR_140"},
        {(CUdevice_attribute)145, "ATTR_145"},
        {(CUdevice_attribute)150, "ATTR_150"},
        {(CUdevice_attribute)160, "ATTR_160"},
        {(CUdevice_attribute)170, "ATTR_170"},
        {(CUdevice_attribute)180, "ATTR_180"},
        {(CUdevice_attribute)190, "ATTR_190"},
        {(CUdevice_attribute)200, "ATTR_200"},
    };

    int nattrs = sizeof(attrs) / sizeof(attrs[0]);
    for (int i = 0; i < nattrs; i++)
    {
        int val;
        CUresult res = cuDeviceGetAttribute(&val, attrs[i].attr, cudev);
        if (res == CUDA_SUCCESS && val != 0)
        {
            printf("    %-45s = %d", attrs[i].name, val);
            if (val > 0 && strstr(attrs[i].name, "unknown"))
            {
                printf("  <<< NON-ZERO UNKNOWN ATTRIBUTE!");
            }
            printf("\n");
        }
    }

    // Полный скан всех атрибутов 0-300
    printf("\n  Full attribute scan (non-zero only):\n");
    for (int a = 0; a <= 300; a++)
    {
        int val;
        CUresult res = cuDeviceGetAttribute(&val, (CUdevice_attribute)a, cudev);
        if (res == CUDA_SUCCESS && val != 0)
        {
            printf("    attr[%3d] = %d\n", a, val);
        }
    }
}

// ============================================================================
// Test 4: Try loading sm_90 PTX on sm_89 device
// If it loads — the device MIGHT support sm_90 instructions
// ============================================================================

void test_sm90_ptx_load(void)
{
    printf("\n=== sm_90 PTX Load Test ===\n\n");

    CUmodule mod;
    CUjit_option opts[] = {CU_JIT_TARGET};
    void *vals[] = {(void *)(uintptr_t)CU_TARGET_COMPUTE_90};

    // Попробуем загрузить sm_90 PTX
    CUresult res = cuModuleLoadDataEx(&mod, tma_test_ptx, 1, opts, vals);

    if (res == CUDA_SUCCESS)
    {
        printf("  [!!!] sm_90 PTX ЗАГРУЗИЛСЯ на sm_89!\n");
        printf("  Это может означать:\n");
        printf("  - JIT компилятор понизил до sm_89 (обычное поведение)\n");
        printf("  - Или device поддерживает sm_90 инструкции\n");

        CUfunction func;
        res = cuModuleGetFunction(&func, mod, "tma_kernel");
        if (res == CUDA_SUCCESS)
        {
            printf("  [!!!] tma_kernel функция НАЙДЕНА!\n");

            // Проверим атрибуты функции
            int numRegs, smemSize;
            cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
            cuFuncGetAttribute(&smemSize, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
            printf("  Registers: %d, Shared mem: %d\n", numRegs, smemSize);
        }

        cuModuleUnload(mod);
    }
    else
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("  sm_90 PTX НЕ загрузился: %s\n", errStr);
        printf("  Это ожидаемо если TMA hardware отсутствует\n");
    }

    // Попробуем sm_89 PTX с cp.async (должен работать)
    const char *cp_async_ptx =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry async_test(.param .u64 src, .param .u64 dst) {\n"
        "    .reg .b64 %rd<4>;\n"
        "    .reg .b32 %r<4>;\n"
        "    .shared .align 16 .b8 smem[256];\n"
        "    ld.param.u64 %rd0, [src];\n"
        "    mov.u32 %r0, %tid.x;\n"
        "    shl.b32 %r1, %r0, 2;\n"
        "    add.u64 %rd1, %rd0, %r1;\n"
        "    // cp.async — supported on sm_89\n"
        "    cp.async.ca.shared.global [smem + %r1], [%rd1], 4;\n"
        "    cp.async.commit_group;\n"
        "    cp.async.wait_group 0;\n"
        "    ret;\n"
        "}\n";

    res = cuModuleLoadData(&mod, cp_async_ptx);
    if (res == CUDA_SUCCESS)
    {
        printf("\n  sm_89 cp.async PTX: ЗАГРУЗИЛСЯ ✓ (ожидаемо)\n");
        cuModuleUnload(mod);
    }
    else
    {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        printf("\n  sm_89 cp.async PTX: ОШИБКА: %s\n", errStr);
    }
}

// ============================================================================
// Test 5: cp.async capability test — проверяем максимальный размер
// На Hopper cp.async.bulk поддерживает до 256B за раз
// На Ada cp.async — 4/8/16 bytes
// Если Ada поддерживает >16B — может быть скрытый TMA
// ============================================================================

__global__ void test_cp_async_16b(float4 *dst, const float4 *src)
{
    extern __shared__ float4 smem4[];
    int tid = threadIdx.x;

    // cp.async 16 bytes (максимум для sm_80-89)
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)(uintptr_t)&smem4[tid]),
        "l"(&src[blockIdx.x * blockDim.x + tid]));

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    dst[blockIdx.x * blockDim.x + tid] = smem4[tid];
}

void test_cp_async_sizes(void)
{
    printf("\n=== cp.async Size Capability Test ===\n\n");

    float4 *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, 1024 * sizeof(float4)));
    CHECK(cudaMalloc(&d_dst, 1024 * sizeof(float4)));

    // Заполним данные
    float4 *h_src = (float4 *)malloc(1024 * sizeof(float4));
    for (int i = 0; i < 1024; i++)
    {
        h_src[i].x = i * 1.0f;
        h_src[i].y = i * 2.0f;
        h_src[i].z = i * 3.0f;
        h_src[i].w = i * 4.0f;
    }
    CHECK(cudaMemcpy(d_src, h_src, 1024 * sizeof(float4), cudaMemcpyHostToDevice));

    // Test 16B cp.async
    test_cp_async_16b<<<4, 256, 256 * sizeof(float4)>>>(d_dst, d_src);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        CHECK(cudaDeviceSynchronize());
        float4 h_dst[4];
        CHECK(cudaMemcpy(h_dst, d_dst, 4 * sizeof(float4), cudaMemcpyDeviceToHost));
        printf("  cp.async 16B: РАБОТАЕТ ✓ (val[0] = %.1f)\n", h_dst[0].x);
    }
    else
    {
        printf("  cp.async 16B: ОШИБКА: %s\n", cudaGetErrorString(err));
    }

    free(h_src);
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
}

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TMA Instruction & Capability Test — RTX 4090 (sm_89)     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    CHECK_DRV(cuInit(0));

    // 1. Device caps
    query_device_caps();

    // 2. PTX load test
    test_sm90_ptx_load();

    // 3. cp.async sizes
    test_cp_async_sizes();

    printf("\n=== ИТОГ ===\n\n");
    printf("TMA (Tensor Memory Accelerator) на Hopper состоит из:\n");
    printf("  1. SM pipeline decoder для cp.async.bulk.tensor опкодов\n");
    printf("  2. Tensor descriptor hardware (создание/интерпретация)\n");
    printf("  3. CTA-level DMA unit (асинхронная bulk загрузка)\n");
    printf("  4. Barrier hardware extension (mbarrier arrive/wait)\n\n");
    printf("На Ada (sm_89) есть:\n");
    printf("  • cp.async (до 16B) — базовый async pipeline\n");
    printf("  • Shared memory barrier — __syncthreads уровень\n");
    printf("  • НЕТ tensor descriptors\n");
    printf("  • НЕТ bulk transfer (>16B за одну операцию)\n\n");
    printf("Проверь MMIO результаты (tma_probe) для hardware evidence.\n");

    return 0;
}
