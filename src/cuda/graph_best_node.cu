
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include <driveless/cuda_ptr.h>
#include "../../include/graph.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);

#define K1 1
#define K2 3
#define K3 1

__device__ long long __compute_cost_findBestNode(float dist, float heading_rad, float nodeCost) {
    return __float2ll_rd(K1 * dist + K2 * TO_DEG * heading_rad + K3 * nodeCost);
}


__global__ void __CUDA_KERNEL_findBestNodeWithHeading_bestCost(
    int4 *graph,
    float4 *graphData,
    float3 *frame,
    int *params,
    float *classCost,
    float searchRadius,
    int targetX,
    int targetZ,
    float targetHeading_rad,
    float maxHeadingError_rad,
    long long *bestCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int minDistX = params[FRAME_PARAM_MIN_DIST_X];
    int minDistZ = params[FRAME_PARAM_MIN_DIST_Z];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
        return;

    int dx = targetX - x;
    int dz = targetZ - z;

    const float dist = sqrtf(dx * dx + dz * dz);

    if (dist > searchRadius) {
        //printf ("%d, %d failed because of dist\n", x, z);
        return;
    }

    float heading = getHeadingCuda(graphData, pos);

    if (abs(heading - targetHeading_rad) > maxHeadingError_rad) {
        //printf ("%d, %d failed because of heading error: %f vs %f\n", x, z, abs(heading - targetHeading_rad), maxHeadingError_rad);
        return;
    }

    if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, heading)) {
        //printf ("%d, %d, %f failed because is unfeasible minDistX = %d, minDistZ = %d\n", x, z, heading, minDistX, minDistZ);
        return;
    }

    long long cost = __compute_cost_findBestNode(dist, heading, getCostCuda(graphData, pos));

    atomicMin(bestCost, cost);
}

__global__ void __CUDA_KERNEL_findBestNodeWithHeading_firstNodeWithCost(
    int4 *graph, 
    float4 *graphData, 
    float3 *frame, 
    int *params, 
    float *classCost, 
    float searchRadius, 
    int targetX, 
    int targetZ, 
    float targetHeading_rad, 
    float maxHeadingError_rad,
    long long bestCost, 
    int2 *node)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int minDistX = params[FRAME_PARAM_MIN_DIST_X];
    int minDistZ = params[FRAME_PARAM_MIN_DIST_Z];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
        return;

    int dx = targetX - x;
    int dz = targetZ - z;

    const float dist = sqrtf(dx * dx + dz * dz);

    if (dist > searchRadius)
        return;

    float heading = getHeadingCuda(graphData, pos);

    if (abs(heading - targetHeading_rad) > maxHeadingError_rad)
        return;

    if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, heading))
    {
        return;
    }

    long long cost = __compute_cost_findBestNode(dist, heading, getCostCuda(graphData, pos));

    if (cost == bestCost)
    {
        (*node).x = x;
        (*node).y = z;
    }
}

int2 CudaGraph::findBestNode(float3 *og, angle heading, float radius, int x, int z, float maxHeadingError)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    CudaPtr<int2> bestNode(1);
    CudaPtr<long long> cost(1);

    bestNode.get()->x = -1;
    bestNode.get()->y = -1;
    *cost.get() = 99999999999;

    __CUDA_KERNEL_findBestNodeWithHeading_bestCost<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _searchSpaceParams->get(),
        _classCosts->get(),
        radius,
        x, z, heading.rad(), 
        maxHeadingError,
        cost.get());

    CUDA(cudaDeviceSynchronize());

    if (*cost.get() >= 99999999999)
        return {-1, -1};

    __CUDA_KERNEL_findBestNodeWithHeading_firstNodeWithCost<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _searchSpaceParams->get(),
        _classCosts->get(),
        radius,
        x, z, 
        heading.rad(),
        maxHeadingError,
        *cost.get(),         
        bestNode.get());

    CUDA(cudaDeviceSynchronize());

    return { bestNode.get()->x,  bestNode.get()->y};
}

extern __device__ __host__ double compute_euclidean_2d_dist(const int2 &start, const int2 &end);

__global__ void __CUDA_KERNEL_checkGoalReached(
    int4 *graph, 
    float4 *graphData, 
    float3 *frame, 
    int *params, 
    float *costs, 
    int goalX, 
    int goalZ, 
    float goalHeading, 
    float distToGoalTolerance, 
    float maxHeadingError,
    bool *goalReached)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (graph[pos].z != GRAPH_TYPE_NODE)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int2 s = {x, z};
    int2 e = {goalX, goalZ};

    if (compute_euclidean_2d_dist(s, e) > distToGoalTolerance)
        return;
    
    float heading = getHeadingCuda(graphData, pos);

    if (abs(heading - goalHeading) <= maxHeadingError)
        *goalReached = true;

    //printf ("goal reached candidate: %d, %d --> %d, %d,  dist: %f < %f heading error: %f, max: %f\n", x, z, goalX, goalZ, compute_euclidean_2d_dist(s, e), distToGoalTolerance, abs(heading - goalHeading), maxHeadingError);

}

bool CudaGraph::checkGoalReached(float3 *og, int2 goal, angle heading, float distanceToGoalTolerance, float maxHeadingError)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    if (*_goalReached->get())
        return true;

     //printf("check goal: %d, %d\n", goal.x, goal.y);

    __CUDA_KERNEL_checkGoalReached<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _searchSpaceParams->get(),
        _classCosts->get(),
        goal.x,
        goal.y,
        heading.rad(),
        distanceToGoalTolerance,
        maxHeadingError,
        _goalReached->get());

    CUDA(cudaDeviceSynchronize());

    //printf ("goal checked\n");
    return *_goalReached->get();
}