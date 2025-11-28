
#include "../../include/graph.h"
#include <driveless/cuda_params.h>
#include <bits/algorithmfwd.h>

extern __device__ __host__ float checkDirectConnectionToGoal(float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading, bool isSafeZoneChecked, bool isDistanceToGoalProcessed);
extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);

/// @brief Computes a direct Hermite connection to all other forward nodes in the path and stores the lowest cost in path_optim_data[path_pos] as (1, x, z, cost) - 1 means lower cost found
/// @param graph
/// @param graphData
/// @param frame
/// @param directConnectData
/// @param classCosts
/// @param searchSpaceParams
/// @param path
/// @param path_optim_data
/// @param path_pos
/// @param path_size
/// @param max_curvature
/// @param isSafeZoneChecked
/// @param isDistanceToGoalProcessed
/// @return
__device__ __host__ void __check_direct_connection_to_forward_nodes(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams,
                                                                    float4 *path, float4 *path_optim_data, int path_pos, int path_size, float max_curvature, bool isSafeZoneChecked, bool isDistanceToGoalProcessed)
{
    if (path_pos >= path_size - 2)
        return;

    float local_heading = path[path_pos].z;

    path_optim_data[path_pos].x = 0;

    int x = TO_INT(path[path_pos].x);
    int z = TO_INT(path[path_pos].y);

    for (int i = path_pos + 2; i < path_size; i++)
    {
        float4 nextp = path[i];
        float currentCost = path[i].w;

        float newCost = checkDirectConnectionToGoal(graphData, frame, classCosts, searchSpaceParams, max_curvature, x, z,
                                                    local_heading, TO_INT(nextp.x), TO_INT(nextp.y), nextp.z, isSafeZoneChecked, isDistanceToGoalProcessed);

        //printf("checking %d, %d --> %d, %d - current cost: %f, new cost: %f\n", x, z, TO_INT(nextp.x), TO_INT(nextp.y), currentCost, newCost);

        if (newCost > 0 && newCost < currentCost)
        {
            path_optim_data[path_pos].x = 1;
            path_optim_data[path_pos].y = i;
            path_optim_data[path_pos].z = newCost;
            path_optim_data[path_pos].w = currentCost - newCost;
        }
    }
}

// __global__ void __check_direct_connection_on_all_path_nodes(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams,
//                                                             float4 *path, float4 *path_optim_data, int path_pos, int path_size, float max_curvature, bool isSafeZoneChecked, bool isDistanceToGoalProcessed)
// {
//     int pos = blockIdx.x * blockDim.x + threadIdx.x;

//     const int width = searchSpaceParams[FRAME_PARAM_WIDTH];
//     const int height = searchSpaceParams[FRAME_PARAM_HEIGHT];

//     if (pos >= width * height)
//         return;

//     if (getTypeCuda(graph, pos) != GRAPH_TYPE_NODE)
//         return;

//     int z = pos / width;
//     int x = pos - z * width;
// }

sptr<float4> CudaGraph::convertPlannedPath(std::vector<Waypoint> path)
{
    sptr<float4> res = std::make_shared<CudaPtr<float4>>(path.size());
    int pos = 0;
    for (auto p : path)
    {
        res->get()[pos].x = path[pos].x();
        res->get()[pos].y = path[pos].z();
        res->get()[pos].z = path[pos].heading().rad();
        res->get()[pos].w = getCost(path[pos].x(), path[pos].z());
        // printf ("path %d, %d pos %d, cost %f\n", path[pos].x(), path[pos].z(), pos, res->get()[pos].w);
        pos++;
    }
    return res;
}

bool CudaGraph::optimizePathLoop(float3 *frame, sptr<float4> path, int path_size, float distanceToGoalTolerance, bool isSafeZoneChecked)
{
    cptr<float4> pathData = std::make_unique<CudaPtr<float4>>(path_size);
    float max_curvature = _physicalParams->get()[PHYSICAL_MAX_CURVATURE];

    for (int i = 0; i < path_size - 2; i++)
    {
        __check_direct_connection_to_forward_nodes(_graph->getCudaPtr(), _graphData->getCudaPtr(), frame, _classCosts->get(),
                                                   _searchSpaceParams->get(), path->get(), pathData->get(), i, path_size, max_curvature, isSafeZoneChecked, false);

        int nextPos = TO_INT(pathData->get()[i].y);
        // if (pathData->get()[i].x == 1.0)
        // {
        //     printf("best gain for node %d, %d connected to %d, %d with cost %f is to re-connected to %d, %d with new cost %f (gain %f)\n",
        //            TO_INT(path->get()[i].x),
        //            TO_INT(path->get()[i].y),
        //            TO_INT(path->get()[i + 1].x),
        //            TO_INT(path->get()[i + 1].y),
        //            path->get()[i + 1].w,
        //            TO_INT(path->get()[nextPos].x),
        //            TO_INT(path->get()[nextPos].y),
        //            pathData->get()[i].z,
        //            pathData->get()[i].w);
        // }
        // else
        // {
        //     printf("node %d, %d must remain connected to %d, %d with cost %f\n", TO_INT(path->get()[i].x),
        //            TO_INT(path->get()[i].y),
        //            TO_INT(path->get()[i + 1].x),
        //            TO_INT(path->get()[i + 1].y),
        //            path->get()[i + 1].w);
        // }
    }

    float maxGain = -1;
    int maxGainPos = -1;
    for (int i = 0; i < path_size - 2; i++)
    {
        float4 p = pathData->get()[i];
        if (p.x == 0.0)
            continue;

        if (maxGain < p.w)
        {
            maxGain = p.w;
            maxGainPos = i;
        }
    }

    if (maxGainPos < 0) return false;

    float4 p = pathData->get()[maxGainPos];
    int next_pos = TO_INT(p.y);
    int next_x = path->get()[next_pos].x;
    int next_z = path->get()[next_pos].y;
    // printf("best gain: is to change %d, %d connected to %d, %d with cost %f, reconnecting to %d, %d with new cost %f\n",
    //        TO_INT(path->get()[maxGainPos].x),
    //        TO_INT(path->get()[maxGainPos].y),
    //        TO_INT(path->get()[maxGainPos + 1].x),
    //        TO_INT(path->get()[maxGainPos + 1].y),
    //        path->get()[maxGainPos + 1].w,
    //        next_x, next_z, p.z);

    if (maxGainPos != -1 && next_pos > maxGainPos + 1) {
        
        // storing the best cost for the last node before removal/path rewrite
        path->get()[next_pos].w = pathData->get()[maxGainPos].z;

        // Remove elements between maxGainPos and next_pos (exclusive)
        // Shift elements left
        int numToRemove = next_pos - maxGainPos - 1;
        for (int i = maxGainPos + 1; i + numToRemove < path_size; ++i) {
            path->get()[i] = path->get()[i + numToRemove];
        }
        path_size -= numToRemove;
    }

    clear();

    float4 parent = path->get()[0];
    int parent_x = TO_INT(parent.x);
    int parent_z = TO_INT(parent.y);
    add(parent_x, parent_z, angle::rad(parent.z), -1, -1, parent.w);

    for (int i = 1; i < path_size; i++) {
        float4 child = path->get()[i];
        add(TO_INT(child.x), TO_INT(child.y), angle::rad(child.z), parent_x, parent_z, child.w);
        parent = child;
        parent_x = TO_INT(parent.x);
        parent_z = TO_INT(parent.y);
    }

    return true;

    // // Print the path in the requested format
    // printf("(");
    // for (int i = 0; i < path_size; ++i) {
    //     printf("%d, %d", TO_INT(path->get()[i].x), TO_INT(path->get()[i].y));
    //     if (i < path_size - 1) {
    //         printf(") --> (");
    //     }
    // }
    // printf(")\n");
}

// int size = _graph->width() * _graph->height();
// int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

//     __CUDA_check_direct_connection_forward_nodes<<<numBlocks, THREADS_IN_BLOCK>>> (
//         ,
//     )
// CUDA(cudaDeviceSynchronize());
