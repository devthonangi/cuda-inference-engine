#include "cuda_kernels.cuh"

__global__ void reluKernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = fmaxf(0.0f, data[idx]);
}

void launchReLU(float* d_data, int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d_data, size);
    cudaDeviceSynchronize();
}
