#include <cstdio>
#include <cuda_runtime.h>

extern __shared__ char smem[];

__global__ void test_kernel(int size)
{
    if (threadIdx.x == 0)
    {
        for (int i = size - 256; i < size; i++)
            smem[i] = (char)(i & 0xFF);
        int ok = 1;
        for (int i = size - 256; i < size; i++)
            if (smem[i] != (char)(i & 0xFF))
                ok = 0;
        if (ok)
            printf("  %d KB: WORKS\n", size / 1024);
        else
            printf("  %d KB: CORRUPT\n", size / 1024);
    }
}

int main()
{
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Reported max shared/SM: %zu\n", p.sharedMemPerMultiprocessor);
    printf("Reported max shared/block: %zu\n", p.sharedMemPerBlock);

    int sizes[] = {100, 101, 102, 104, 112, 120, 128, 144, 160, 192, 228, 256};
    for (int kb : sizes)
    {
        int bytes = kb * 1024;
        cudaError_t err = cudaFuncSetAttribute(test_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err != cudaSuccess)
        {
            printf("  %d KB: SetAttribute REJECTED (%s)\n", kb, cudaGetErrorString(err));
            continue;
        }
        test_kernel<<<1, 32, bytes>>>(bytes);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("  %d KB: LAUNCH FAILED (%s)\n", kb, cudaGetErrorString(err));
            cudaGetLastError();
        }
    }
    return 0;
}
