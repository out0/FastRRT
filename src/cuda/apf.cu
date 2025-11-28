
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../../include/graph.h"
#include <math_constants.h>

#define FORCE_RANGE 5

extern __device__ __host__ void incIntrinsicCost(float4 *graphData, int width, int x, int z, float cost);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getIntrinsicCostCuda(float4 *graphData, long pos);
extern __device__ __host__ void setIntrinsicCostCuda(float4 *graphData, long pos, float cost);

__device__ inline bool in_range(int width, int height, int x, int z) {
    return x >= 0 && x < width && z >= 0 && z < height;
}

__device__ inline bool check_is_obstacle(float3 *og, float *classCosts, int width, int height, int x, int z) {
    if (!in_range(width, height, x, z)) return true;
    long pos = computePos(width, x, z);
    return classCosts[TO_INT(og[pos].x)] < 0;
}

__global__ static void __CUDA_KERNEL_repulsive_force(float3 *og, float4 *graphData, float *classCosts, int *params, int width, int height, float Kr_half, int radius)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    // if (x == 46 && z == 46) {
    //     printf ("(%d, %d) classCost = %f\n", x, z);
    // }

    float c = classCosts[TO_INT(og[pos].x)];
    if (c >= 0)
        return;  // not an obstacle

    setIntrinsicCostCuda(graphData, pos, 100*Kr_half);

    if (check_is_obstacle(og, classCosts, width, height, x - 1, z) &&
        check_is_obstacle(og, classCosts, width, height, x + 1, z) &&
        check_is_obstacle(og, classCosts, width, height, x, z - 1) &&
        check_is_obstacle(og, classCosts, width, height, x, z + 1))
        return;

    for (int h = z - radius; h <= z + radius; h++)
    {
        if (h < 0)
            continue;
        
        if (h >= height)
            break;

        int init = x - radius;
        float p0 = (float)radius;

        for (int w = init; w <= x + radius; w++)
        {
            if (w < 0)
                continue;

            if (w >= width)
                break;

            if (w >= lower_bound_ego_x && w <= upper_bound_ego_x && h >= upper_bound_ego_z && h <= lower_bound_ego_z)
                continue;
            
            if (w == x && h == z)
                continue;
            

            float p = sqrtf((w - x) * (w - x) + (h - z) * (h - z));
            if (p > p0)
                continue;

            float f = 1/p - 1/p0;
            float cost = Kr_half * f * f;
            incIntrinsicCost(graphData, width, w, h, cost);
            //printf("(%d, %d): r = %f, cost =  %f, 1/p = %f, 1/p0 = %f, f = %f\n", w, h, r, cost, 1/p, 1/p0, f);

            // printf("(%d, %d) cost = %f\n", w, h, cost);
        }
    }
}

__global__ static void __CUDA_KERNEL_attractive_force(float3 *og, float4 *graphData, float *classCosts, int width, int height, float Ka_half, int goal_x, int goal_z)
{
    long pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    float c = classCosts[TO_INT(og[pos].x)];
    if (c < 0) {
        return;  // obstacle
    }


    int dx = goal_x - x;
    int dz = goal_z - z;

    float dcost = (float)(dx * dx + dz * dz) * Ka_half;

    float current_cost = getIntrinsicCostCuda(graphData, pos);
    setIntrinsicCostCuda(graphData, pos, current_cost - dcost);
}

void CudaGraph::computeRepulsiveFieldAPF(float3 *og, float Kr, int radius)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_repulsive_force<<<numBlocks, THREADS_IN_BLOCK>>>(
        og,
        _graphData->getCudaPtr(),
        _classCosts->get(),
        _searchSpaceParams->get(),
        _graph->width(),
        _graph->height(),
        Kr * 0.5,
        radius);

    CUDA(cudaDeviceSynchronize());
}
void CudaGraph::computeAttractiveFieldAPF(float3 *og, float Ka, std::pair<int, int> goal) {
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_attractive_force<<<numBlocks, THREADS_IN_BLOCK>>>(
        og,
        _graphData->getCudaPtr(),
        _classCosts->get(),
        _graph->width(),
        _graph->height(),
        Ka * 0.5,
        goal.first,
        goal.second);

    CUDA(cudaDeviceSynchronize());
}
