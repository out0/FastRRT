#include <curand_kernel.h>
#include <chrono>
#include "../../include/graph.h"

__global__ void __CUDA_KERNEL_setupRandomGenKernel(curandState *state, int size, long long seed){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__device__ float generateRandom(curandState* state, int pos, float min_val, float max_val) {
    return max(min_val, max_val * curand_uniform(&state[pos]));
}

__device__ float generateRandomNeg(curandState* state, int pos, float max_val) {
    return  (2.0f * curand_uniform(&state[pos]) - 1.0f) * max_val;
}

    
long long getCurrentTimeMillis() {
    // Get current time_point
    auto now = std::chrono::system_clock::now();
    // Convert to duration since epoch
    auto duration = now.time_since_epoch();
    // Convert duration to milliseconds and return as long long
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void CudaGraph::__initializeRandomGenerator()
{   
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    _randState = std::make_unique<CudaPtr<curandState>>(_graph->width() * _graph->height());

    __CUDA_KERNEL_setupRandomGenKernel<<<numBlocks, THREADS_IN_BLOCK>>>(
        _randState->get(), size, getCurrentTimeMillis());

    CUDA(cudaDeviceSynchronize());

    srand(time(0));
}