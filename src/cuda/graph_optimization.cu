
#include "../../include/cuda_graph.h"
#include <driveless/frame_params.h>
#include <bits/algorithmfwd.h>

extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ float4 checkDirectConnectionToGoal(float4 *graphData, float3 *frame,
                                                              float *classCosts, int *searchSpaceParams, float max_curvature,
                                                              int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading,
                                                              bool isSafeZoneChecked, bool isDistanceToGoalProcessed,
                                                              float distance_to_goal_tolerance,
                                                              float max_heading_error);

extern __device__ __host__ bool preProcessedCollisionDistance(int *searchParams);
extern __device__ __host__ bool preProcessedCollisionVector(int *searchParams);
extern __device__ __host__ bool preProcessedDistanceToGoal(int *searchParams);

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

        float4 newCost = checkDirectConnectionToGoal(graphData,
                                                     frame,
                                                     classCosts,
                                                     searchSpaceParams,
                                                     max_curvature,
                                                     x, z,
                                                     local_heading,
                                                     TO_INT(nextp.x),
                                                     TO_INT(nextp.y),
                                                     nextp.z,
                                                     isSafeZoneChecked,
                                                     isDistanceToGoalProcessed,
                                                     0, 0);

        // printf("checking %d, %d --> %d, %d - current cost: %f, new cost: %f\n", x, z, TO_INT(nextp.x), TO_INT(nextp.y), currentCost, newCost);

        if (newCost.x > 0 && newCost.w < currentCost)
        {
            path_optim_data[path_pos].x = 1;
            path_optim_data[path_pos].y = i;
            path_optim_data[path_pos].z = newCost.w;
            path_optim_data[path_pos].w = currentCost - newCost.w;
        }
    }
}

#ifdef DRIVELESS_CUDA_ENABLED
sptr<float4> CudaGraph::convertPlannedPath(std::vector<Waypoint> path) {
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
#else
std::shared_ptr<float4[]> CudaGraph::convertPlannedPath(std::vector<Waypoint> path)
{
    std::shared_ptr<float4[]> res(new float4[path.size()], std::default_delete<float4[]>());
    int pos = 0;
    for (auto p : path)
    {
        res.get()[pos].x = path[pos].x();
        res.get()[pos].y = path[pos].z();
        res.get()[pos].z = path[pos].heading().rad();
        res.get()[pos].w = getCost(path[pos].x(), path[pos].z());
        // printf ("path %d, %d pos %d, cost %f\n", path[pos].x(), path[pos].z(), pos, res->get()[pos].w);
        pos++;
    }
    return res;
}
#endif

#ifdef DRIVELESS_CUDA_ENABLED
bool CudaGraph::optimizePathLoop(float3 *frame, sptr<float4> path, int path_size, float distanceToGoalTolerance)
#else
bool CudaGraph::optimizePathLoop(float3 *frame, std::shared_ptr<float4[]> path, int path_size, float distanceToGoalTolerance)
#endif
{
#ifdef DRIVELESS_CUDA_ENABLED
    cptr<float4> pathData = std::make_unique<CudaPtr<float4>>(path_size);
    xxx const float max_curvature = _physicalParams->get()[PHYSICAL_MAX_CURVATURE];
    const bool preProcessCollisionDistance = preProcessedCollisionDistance(_searchSpaceParams->get());
    const bool preProcessDistanceToGoal = preProcessedDistanceToGoal(_searchSpaceParams->get());
#else
    float max_curvature = _physicalParams.get()[PHYSICAL_MAX_CURVATURE];
    const bool preProcessCollisionDistance = preProcessedCollisionDistance(_searchSpaceParams.get());
    const bool preProcessDistanceToGoal = preProcessedDistanceToGoal(_searchSpaceParams.get());
    std::shared_ptr<float4[]> pathData(new float4[path_size], std::default_delete<float4[]>());
#endif

    for (int i = 0; i < path_size - 2; i++)
    {
#ifdef DRIVELESS_CUDA_ENABLED
        __check_direct_connection_to_forward_nodes(
            _graph->getPtr(),
            _graphData->getPtr(),
            frame,
            _classCosts->get(),
            _searchSpaceParams->get(),
            path->get(),
            pathData->get(),
            i,
            path_size,
            max_curvature,
            preProcessCollisionDistance,
            preProcessDistanceToGoal);

        int nextPos = TO_INT(pathData->get()[i].y);
#else
        __check_direct_connection_to_forward_nodes(
            _graph->getPtr(),
            _graphData->getPtr(),
            frame,
            _classCosts.get(),
            _searchSpaceParams.get(),
            path.get(),
            pathData.get(),
            i,
            path_size,
            max_curvature,
            preProcessCollisionDistance,
            preProcessDistanceToGoal);

        int nextPos = TO_INT(pathData.get()[i].y);
#endif
    }

    float maxGain = -1;
    int maxGainPos = -1;
    for (int i = 0; i < path_size - 2; i++)
    {
#ifdef DRIVELESS_CUDA_ENABLED
        float4 p = pathData->get()[i];
#else
        float4 p = pathData.get()[i];
#endif
        if (p.x == 0.0)
            continue;

        if (maxGain < p.w)
        {
            maxGain = p.w;
            maxGainPos = i;
        }
    }

    if (maxGainPos < 0)
        return false;

#ifdef DRIVELESS_CUDA_ENABLED
    float4 p = pathData->get()[maxGainPos];
    int next_pos = TO_INT(p.y);
    int next_x = path->get()[next_pos].x;
    int next_z = path->get()[next_pos].y;
#else
    float4 p = pathData.get()[maxGainPos];
    int next_pos = TO_INT(p.y);
    int next_x = path.get()[next_pos].x;
    int next_z = path.get()[next_pos].y;
#endif

    if (maxGainPos != -1 && next_pos > maxGainPos + 1)
    {
#ifdef DRIVELESS_CUDA_ENABLED
        // storing the best cost for the last node before removal/path rewrite
        path->get()[next_pos].w = pathData->get()[maxGainPos].z;
#else
        path.get()[next_pos].w = pathData.get()[maxGainPos].z;
#endif

        // Remove elements between maxGainPos and next_pos (exclusive)
        // Shift elements left
        int numToRemove = next_pos - maxGainPos - 1;
        for (int i = maxGainPos + 1; i + numToRemove < path_size; ++i)
        {
#ifdef DRIVELESS_CUDA_ENABLED
            path->get()[i] = path->get()[i + numToRemove];
#else
            path.get()[i] = path.get()[i + numToRemove];
#endif
        }
        path_size -= numToRemove;
    }

    clear();

#ifdef DRIVELESS_CUDA_ENABLED
    float4 parent = path->get()[0];
#else
    float4 parent = path.get()[0];
#endif
    int parent_x = TO_INT(parent.x);
    int parent_z = TO_INT(parent.y);
    add(parent_x, parent_z, angle::rad(parent.z), -1, -1, parent.w);

    for (int i = 1; i < path_size; i++)
    {
#ifdef DRIVELESS_CUDA_ENABLED
        float4 child = path->get()[i];
#else
        float4 child = path.get()[i];
#endif
        add(TO_INT(child.x), TO_INT(child.y), angle::rad(child.z), parent_x, parent_z, child.w);
        parent = child;
        parent_x = TO_INT(parent.x);
        parent_z = TO_INT(parent.y);
    }

    return true;
}
