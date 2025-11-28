
#include "../../include/graph.h"
#include <driveless/cuda_params.h>
#include <bits/algorithmfwd.h>

extern __device__ __host__ float checkDirectConnectionToGoal(float4 *graphData,
                                                             float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature,
                                                             int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading,
                                                             bool isSafeZoneChecked, bool isDistanceToGoalProcessed);

extern __device__ __host__ float interpolateNodesUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, int goal_x, int goal_z, float goal_heading);
extern __device__ __host__ float canConnectToGoalUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, int goal_x, int goal_z, float goal_heading);


std::tuple<int, float> CudaGraph::__findFirstDirectConnectionToPos(float3 *og, std::vector<float4> res, int pos, bool isSafeZoneChecked)
{
    if (pos <= 1)
        return {-1, -1};
    float max_curvature = _physicalParams->get()[PHYSICAL_MAX_CURVATURE];

    float4 goal = res[pos];

    for (int i = 0; i < pos - 1; i++)
    {
        float4 start = res[i];
        float newCost = checkDirectConnectionToGoal(
            _graphData->getCudaPtr(), og, _classCosts->get(), _searchSpaceParams->get(), max_curvature,
            static_cast<int>(start.x), static_cast<int>(start.y), start.z,
            static_cast<int>(goal.x), static_cast<int>(goal.y), goal.z,
            isSafeZoneChecked, false);

        if (newCost > 0 && newCost < start.w)
            return {i, newCost};
    }
    return {-1, -1};
}

std::vector<float4> CudaGraph::__getPlannedPath(float3 *og, int2 goal, angle goalHeading, float distanceToGoalTolerance)
{
    std::vector<float4> res;

    // res.push_back(*_goal);
    int2 n = findBestNode(og, goalHeading, distanceToGoalTolerance, goal.x, goal.y, TO_RAD * 10);

    int i = 0;
    while (n.x != -1 && n.y != -1)
    {
        float4 p = {static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(getHeading(n.x, n.y).rad()), getCost(n.x, n.y)};
        res.push_back(p);
        n = getParent(n.x, n.y);

        if (i++ >= 1000000)
        {
            printf("[ERROR] looping too much (%d, %d) i = %ld\n", n.x, n.y, i);
            res.clear();
            return res;
        }
    }

    std::reverse(res.begin(), res.end());
    return res;
}

void CudaGraph::optimizePath(float3 *og, int2 goal, angle goalHeading, float distanceToGoalTolerance, bool isSafeZoneChecked)
{
    std::vector<float4> res = __getPlannedPath(og, goal, goalHeading, distanceToGoalTolerance);
    int end_pos = res.size() - 1;
    float max_curvature = _physicalParams->get()[PHYSICAL_MAX_CURVATURE];


    while (end_pos >= 0)
    {
        std::tuple<int, float> p = __findFirstDirectConnectionToPos(og, res, end_pos, isSafeZoneChecked);
        int pos = std::get<0>(p);
        float newCost = std::get<1>(p);
        if (pos >= 0)
        {
            res.erase(res.begin() + pos + 1, res.end());
        }

        end_pos = pos;
    }

    clear();
 
    add(static_cast<int>(res[0].x), static_cast<int>(res[0].y), angle::rad(res[0].z), -1, -1, res[0].w);
    printf ("adding %d, %d, %f\n", static_cast<int>(res[0].x), static_cast<int>(res[0].y), res[0].z);
    for (int i = 1; i < res.size(); i++)
    {
        float4 start = res[i-1];
        float4 goal = res[i];
        printf ("interpolating %d, %d --> %d, %d\n", static_cast<int>(start.x), static_cast<int>(start.y), static_cast<int>(goal.x), static_cast<int>(goal.y));
        interpolateNodesUsingHermite(_graph->getCudaPtr(), _graphData->getCudaPtr(), og, 
            _classCosts->get(), _searchSpaceParams->get(), max_curvature,
            static_cast<int>(start.x), static_cast<int>(start.y),
            static_cast<int>(goal.x), static_cast<int>(goal.y), goal.z);
    }
}
