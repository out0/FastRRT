
#include "../../include/graph.h"
#include  <driveless/cuda_params.h>
#include <bits/algorithmfwd.h>

extern __device__ __host__ bool check_graph_connection_with_hermite(
    int4 *graph,
    float4 *graphData,
    float3 *frame,
    double *physicalParams,
    int *params,
    float *classCost,
    int2 start,
    int2 end,
    float velocity_m_s,
    float &path_cost);

__device__ __host__ double computeHeading(int x1, int z1, int x2, int z2)
{
    double dz = z2 - z1;
    double dx = x2 - x1;

    if (dx == 0 && dz == 0)
        return 0;

    double v1 = 0;
    if (dz != 0)
        v1 = atan2f(-dz, dx);
    else
        v1 = atan2f(0, dx);

    return HALF_PI - v1;
}

int CudaGraph::__optimizePathDirectConnect(float3 *og, float distanceToGoalTolerance, float velocity_m_s, std::vector<float4> res)
{

    int optimizablePos = -1;
    int2 goal = {static_cast<int>(res.back().x), static_cast<int>(res.back().y)};
    float goalHeading = res.back().z;
    float pathCost = 0.0;

    for (int i = 0; i < res.size() - 1; i++)
    {
        if (check_graph_connection_with_hermite(
                _graph->getCudaPtr(),
                _graphData->getCudaPtr(),
                og,
                _physicalParams->get(),
                _searchSpaceParams->get(),
                _classCosts->get(),
                {static_cast<int>(res[i].x), static_cast<int>(res[i].y)},
                goal,
                velocity_m_s,
                pathCost))
        {
            optimizablePos = i;
            break;
        }
    }

    if (optimizablePos == -1)
        return optimizablePos;

    res.erase(res.begin() + optimizablePos + 1, res.end());

    clear();

    int2 parent = {-1, -1};
    int i = 0;
    for (; i < res.size(); i++)
    {
        add(static_cast<int>(res[i].x), static_cast<int>(res[i].y), angle::rad(res[i].z), parent.x, parent.y, res[i].w);
        parent.x = static_cast<int>(res[i].x);
        parent.y = static_cast<int>(res[i].y);
    }

    float cost = getCost(res[i - 1].x, res[i - 1].y) + pathCost;
    add(goal.x, goal.y, angle::rad(goalHeading), parent.x, parent.y, cost);

    return optimizablePos;
}

std::vector<float4> CudaGraph::__getPlannedPath(float3 *og, int2 goal, angle goalHeading, float distanceToGoalTolerance)
{
    std::vector<float4> res;

    // res.push_back(*_goal);
    int2 n = findBestNode(og, goalHeading, distanceToGoalTolerance, goal.x, goal.y, TO_RAD * 10);

    while (n.x != -1 && n.y != -1)
    {
        float4 p = {static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(getHeading(n.x, n.y).rad()), getCost(n.x, n.y)};
        res.push_back(p);
        n = getParent(n.x, n.y);
    }

    std::reverse(res.begin(), res.end());
    return res;
}

void CudaGraph::__optimizePath(float3 *og, int2 goal, angle goalHeading, float distanceToGoalTolerance, float velocity_m_s, int directOptimPos)
{
    if (directOptimPos == 0)
        return;

    std::vector<float4> res = __getPlannedPath(og, goal, goalHeading, distanceToGoalTolerance);
    //printf("__optimizePath res = %d\n", res.size());
    if (res.size() < 2)
        return;

    int pos = res.size() - 1;
    if (directOptimPos >= 0)
        pos = min(directOptimPos - 1, pos);

    while (pos > 1)
    {
        int2 start = {static_cast<int>(res[pos - 2].x), static_cast<int>(res[pos - 2].y)};
        int2 end = {static_cast<int>(res[pos].x), static_cast<int>(res[pos].y)};

        float pathCost = 0.0;
        if (check_graph_connection_with_hermite(
                _graph->getCudaPtr(),
                _graphData->getCudaPtr(),
                og,
                _physicalParams->get(),
                _searchSpaceParams->get(),
                _classCosts->get(),
                start,
                end,
                velocity_m_s,
                pathCost))
        {
            if (getCost(start.x, start.y) + pathCost < getCost(end.x, end.y))
            {
                // rewire parent
                setParent(end.x, end.y, start.x, start.y);
                setType(res[pos - 1].x, res[pos - 1].y, GRAPH_TYPE_NULL);
            }
            pos -= 2;
        }
        else
        {
            pos--;
        }
    }
}

void CudaGraph::optimizeGraphInit(float3 *og, int2 goal, angle goalHeading, float distanceToGoalTolerance, float velocity_m_s) {
    //printf("init");
    std::vector<float4> res = __getPlannedPath(og, goal, goalHeading, distanceToGoalTolerance);

    //printf("res = %d\n", res.size());
    _directOptimPos = __optimizePathDirectConnect(og, distanceToGoalTolerance, velocity_m_s, res);
}

void CudaGraph::optimizeGraph(float3 *og, int2 goal, angle goalHeading, float distanceToGoalTolerance, float velocity_m_s)
{
    if (_directOptimPos == -1)
        optimizeGraphInit(og, goal, goalHeading, distanceToGoalTolerance, velocity_m_s);

    //printf("loop optim - _directOptimPos = %d\n", _directOptimPos);
    __optimizePath(og, goal, goalHeading, distanceToGoalTolerance, velocity_m_s, _directOptimPos);
}
