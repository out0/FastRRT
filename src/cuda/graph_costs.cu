
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../../include/graph.h"

extern __device__ __host__ int2 getParentCuda(int4 *graph, long pos);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ float getFrameCostCuda(float3 *frame, float *classCost, long pos) ;


__device__ __host__ float computeCost(float3 *frame, int4 *graph, float4 *graphData, double *physicalParams, float *classCosts, int width, float goalHeading_rad, long nodePos, double distToParent) {
    int2 parent = getParentCuda(graph, nodePos);
    float parentCost = getCostCuda(graphData, computePos(width, parent.x, parent.y));
    float heading_error_perc = abs(goalHeading_rad - getHeadingCuda(graphData, nodePos)) / physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];
    return (getFrameCostCuda(frame, classCosts, nodePos) + distToParent) * (1 + heading_error_perc) + parentCost;
}

// extern std::pair<float3 *, int> drawKinematicIdealPath(double *physicalParams, int width, int2 center, Waypoint goal, float velocity_m_s);

// __global__ void __CUDA_KERNEL_computeIntrinsicCosts(float4 *graphData, int width, int height)
// {
//     int pos = blockIdx.x * blockDim.x + threadIdx.x;

//     if (pos >= width * height)
//         return;

//     if (!checkInGraphCuda(graph, pos))
//         return;

//     int z = pos / width;
//     int x = pos - z * width;

// }

// void CudaGraph::computeCostForIdealPath(Waypoint &goal, float velocity_m_s) {
//     auto res = drawKinematicIdealPath(_physicalParams, width(), _gridCenter, goal, velocity_m_s);

//     float3 *path = res.first;
//     int size = res.second;

//     int size = _graph->width() * _graph->height();
//     int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

//     __CUDA_KERNEL_computeIntrinsicCosts<<<numBlocks, THREADS_IN_BLOCK>>>(
//         _graph->getCudaPtr(),
//         _frameData->getCudaPtr(),
//         searchFrame->getCudaPtr(),
//         searchFrame->getCudaFrameParamsPtr(),
//         searchFrame->getCudaClassCostsPtr());

//     CUDA(cudaDeviceSynchronize());

// }