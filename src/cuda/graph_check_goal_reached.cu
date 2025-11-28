
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include <driveless/cuda_ptr.h>
#include "../../include/graph.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ bool is_directly_connected_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern __device__ __host__ float checkDirectConnectionToGoal(float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading, bool isSafeZoneChecked, bool isDistanceToGoalProcessed);
extern __device__ __host__ float get_heading_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern __device__ __host__ float get_cost_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);

__global__ void __CUDA__check_goal_reached_with_direct_connection_cost(
    int4 *graph,
    float4 *graphData,
    float3 *frame,
    float3 *directConnection,
    int *params,
    float *classCost,
    float searchRadius,
    float max_curvature,
    bool safeZoneChecked,
    long long *bestCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
        return;

    float heading = getHeadingCuda(graphData, pos);

    for (int zp = (z - searchRadius); zp < (z + searchRadius); zp++)
        for (int xp = (x - searchRadius); xp < (x + searchRadius); xp++)
        {
            if (zp < 0 || zp >= height)
                continue;
            if (xp < 0 || xp >= width)
                continue;

            if (!is_directly_connected_to_goal(directConnection, width, xp, zp))
                continue;

            float local_intermediate_heading = get_heading_direct_connection_to_goal(directConnection, width, xp, zp);

            float cost_graph_node_to_precomputed_node_with_connection_to_goal = checkDirectConnectionToGoal(graphData, frame, classCost, params, max_curvature, x, z, heading, xp, zp, local_intermediate_heading, safeZoneChecked, false);

            if (cost_graph_node_to_precomputed_node_with_connection_to_goal < 0)
                continue;

            float total_cost = cost_graph_node_to_precomputed_node_with_connection_to_goal + get_cost_direct_connection_to_goal(directConnection, width, xp, zp);

            long long lcost = __float2ll_rd(100 * total_cost);

            if (atomicMin(bestCost, lcost) != lcost) // it means that the value was replaced
                setTypeCuda(graph, pos, GRAPH_TYPE_PROCESSING);
        }
}

__global__ void __CUDA__check_goal_reached_with_direct_connection(
    int4 *graph,
    float4 *graphData,
    float3 *frame,
    float3 *directConnection,
    int *params,
    float *classCost,
    float searchRadius,
    float max_curvature,
    bool safeZoneChecked,
    long long bestCost,
    float4 *nodes)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_PROCESSING) // w means that the point is part of the graph
        return;

    setTypeCuda(graph, pos, GRAPH_TYPE_NODE);

    float heading = getHeadingCuda(graphData, pos);

    for (int zp = (z - searchRadius); zp < (z + searchRadius); zp++)
        for (int xp = (x - searchRadius); xp < (x + searchRadius); xp++)
        {
            if (zp < 0 || zp >= height)
                continue;
            if (xp < 0 || xp >= width)
                continue;

            if (!is_directly_connected_to_goal(directConnection, width, xp, zp))
                continue;

            float local_intermediate_heading = get_heading_direct_connection_to_goal(directConnection, width, xp, zp);

            float cost_graph_node_to_precomputed_node_with_connection_to_goal = checkDirectConnectionToGoal(graphData, frame, classCost, params, max_curvature, x, z, heading, xp, zp, local_intermediate_heading, safeZoneChecked, false);

            if (cost_graph_node_to_precomputed_node_with_connection_to_goal < 0)
                continue;

            float total_cost = cost_graph_node_to_precomputed_node_with_connection_to_goal + get_cost_direct_connection_to_goal(directConnection, width, xp, zp);

            long long lcost = __float2ll_rd(100 * total_cost);
            // printf ("%d, %d direct connection to goal - %d, %d bestCost: %f cost: %f\n", x, z, xp, zp, bestCost, lcost);
            if (lcost <= bestCost)
            {
                //parent
                nodes[0].x = x;
                nodes[0].y = z;
                nodes[0].z = heading;
                nodes[0].w = cost_graph_node_to_precomputed_node_with_connection_to_goal;
                //child
                nodes[1].x = xp;
                nodes[1].y = zp;
                nodes[1].z = local_intermediate_heading;
                nodes[1].w = total_cost;
            }
        }
}

bool CudaGraph::findBestGoalDirectConnection(float3 *og, float radius, bool isSafeZoneChecked)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    _bestNodeDirectConnection->get()[0].x = -1;
    _bestNodeDirectConnection->get()[0].y = -1;
    *_bestNodeDirectConnectionCost->get() = 99999999999;

    float max_curvature = _physicalParams->get()[PHYSICAL_MAX_CURVATURE];

    __CUDA__check_goal_reached_with_direct_connection_cost<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _graphGoalDirectConnection->getCudaPtr(),
        _searchSpaceParams->get(),
        _classCosts->get(),
        radius,
        max_curvature,
        isSafeZoneChecked,
        _bestNodeDirectConnectionCost->get());

    CUDA(cudaDeviceSynchronize());

    if (*_bestNodeDirectConnectionCost->get() >= 99999999999)
        return false;

    __CUDA__check_goal_reached_with_direct_connection<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _graphGoalDirectConnection->getCudaPtr(),
        _searchSpaceParams->get(),
        _classCosts->get(),
        radius,
        max_curvature,
        isSafeZoneChecked,
        *_bestNodeDirectConnectionCost->get(),
        _bestNodeDirectConnection->get());

    CUDA(cudaDeviceSynchronize());
    return true;

    // float best_cost = *cost.get() / 100;

    // return {(float)bestNode.get()->x, (float)bestNode.get()->y, best_cost};
}

float4 CudaGraph::bestGraphDirectConnectionParent()
{
    return {_bestNodeDirectConnection->get()[0].x, _bestNodeDirectConnection->get()[0].y, _bestNodeDirectConnection->get()[0].z, _bestNodeDirectConnection->get()[0].w};
}

float4 CudaGraph::bestGraphDirectConnectionChild()
{
    return {_bestNodeDirectConnection->get()[1].x, _bestNodeDirectConnection->get()[1].y, _bestNodeDirectConnection->get()[1].z, _bestNodeDirectConnection->get()[1].w};
}