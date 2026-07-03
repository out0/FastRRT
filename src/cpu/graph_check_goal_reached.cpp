
#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include <cmath>
#include "../../include/cuda_graph.h"
#include "atomic_utils.h"
#include <driveless/cpu_parallel_processor.h>

extern bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern float getCostCpu(float4 *graphData, long pos);
extern long computePos(int width, int x, int z);
extern float getHeadingCpu(float4 *graphData, long pos);
extern bool is_directly_connected_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern float checkDirectConnectionToGoal(float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading, bool isSafeZoneChecked, bool isDistanceToGoalProcessed);
extern float get_heading_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern float get_cost_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern void setTypeCpu(int4 *graph, long pos, int type);

class CheckGoalReachedWithDirectConnectionCostProcess : public ParallelProcessor
{
private:
    int4 *_graph;
    float4 *_graphData;
    float3 *_frame;
    float3 *_directConnection;
    int *_params;
    float *_classCost;
    float _searchRadius;
    float _max_curvature;
    bool _safeZoneChecked;
    std::atomic<long long> _bestCost;
    int _max;

public:
    CheckGoalReachedWithDirectConnectionCostProcess(int4 *graph,                               //
                                                    float4 *graphData, float3 *frame,          //
                                                    float3 *directConnection, int *params,     //
                                                    float *classCost, float searchRadius,      //
                                                    float max_curvature, bool safeZoneChecked, //
                                                    int numThreadHandlers = 12) :              //
                                                                                  ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]),
                                                                                  _graph(graph), _graphData(graphData),                       //
                                                                                  _frame(frame), _directConnection(directConnection),         //
                                                                                  _params(params), _classCost(classCost),                     //
                                                                                  _searchRadius(searchRadius), _max_curvature(max_curvature), //
                                                                                  _safeZoneChecked(safeZoneChecked)
    {
        _max = params[FRAME_PARAM_HEIGHT] * params[FRAME_PARAM_WIDTH];
        _bestCost = std::numeric_limits<int>::max();
    }

    void handler(int pos) override
    {

        if (pos >= _max)
            return;

        int width = _params[FRAME_PARAM_WIDTH];
        int height = _params[FRAME_PARAM_HEIGHT];

        int z = pos / width;
        int x = pos - z * width;

        if (_graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
            return;

        float heading = getHeadingCpu(_graphData, pos);

        for (int zp = (z - _searchRadius); zp < (z + _searchRadius); zp++)
            for (int xp = (x - _searchRadius); xp < (x + _searchRadius); xp++)
            {
                if (zp < 0 || zp >= height)
                    continue;
                if (xp < 0 || xp >= width)
                    continue;

                if (!is_directly_connected_to_goal(_directConnection, width, xp, zp))
                    continue;

                float local_intermediate_heading = get_heading_direct_connection_to_goal(_directConnection, width, xp, zp);

                float cost_graph_node_to_precomputed_node_with_connection_to_goal = checkDirectConnectionToGoal(_graphData, _frame, _classCost, _params, _max_curvature, x, z, heading, xp, zp, local_intermediate_heading, _safeZoneChecked, false);

                if (cost_graph_node_to_precomputed_node_with_connection_to_goal < 0)
                    continue;

                float total_cost = cost_graph_node_to_precomputed_node_with_connection_to_goal + get_cost_direct_connection_to_goal(_directConnection, width, xp, zp);

                long long lcost = static_cast<long long>(std::floor(100.0f * total_cost));

                if (atomicMin(_bestCost, lcost) != lcost) // it means that the value was replaced
                    setTypeCpu(_graph, pos, GRAPH_TYPE_PROCESSING);
            }
    }

    long long bestCost()
    {
        return _bestCost;
    }
};

class CheckGoalReachedWithDirectConnectionProcess : public ParallelProcessor
{
private:
    int4 *_graph;
    float4 *_graphData;
    float3 *_frame;
    float3 *_directConnection;
    int *_params;
    float *_classCost;
    float _searchRadius;
    float _max_curvature;
    bool _safeZoneChecked;
    long long _bestCost;
    float4 *_nodes;
    int _max;

public:
    CheckGoalReachedWithDirectConnectionProcess(int4 *graph,                               //
                                                float4 *graphData, float3 *frame,          //
                                                float3 *directConnection, int *params,     //
                                                float *classCost, float searchRadius,      //
                                                float max_curvature, bool safeZoneChecked, //
                                                long long bestCost, float4 *nodes,         //
                                                int numThreadHandlers = 12) :              //
                                                                              ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]),
                                                                              _graph(graph), _graphData(graphData),                       //
                                                                              _frame(frame), _directConnection(directConnection),         //
                                                                              _params(params), _classCost(classCost),                     //
                                                                              _searchRadius(searchRadius), _max_curvature(max_curvature), //
                                                                              _safeZoneChecked(safeZoneChecked), _bestCost(bestCost),     //
                                                                              _nodes(nodes)
    {
        _max = params[FRAME_PARAM_HEIGHT] * params[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int W = _params[FRAME_PARAM_WIDTH];
        int H = _params[FRAME_PARAM_HEIGHT];

        int z = pos / W;
        int x = pos - z * W;

        if (_graph[pos].z != GRAPH_TYPE_PROCESSING) // w means that the point is part of the graph
            return;

        setTypeCpu(_graph, pos, GRAPH_TYPE_NODE);

        float heading = getHeadingCpu(_graphData, pos);

        for (int zp = (z - _searchRadius); zp < (z + _searchRadius); zp++)
        {
            for (int xp = (x - _searchRadius); xp < (x + _searchRadius); xp++)
            {
                if (zp < 0 || zp >= H)
                    continue;
                if (xp < 0 || xp >= W)
                    continue;

                if (!is_directly_connected_to_goal(_directConnection, W, xp, zp))
                    continue;

                float local_intermediate_heading = get_heading_direct_connection_to_goal(_directConnection, W, xp, zp);

                float cost_graph_node_to_precomputed_node_with_connection_to_goal = checkDirectConnectionToGoal(_graphData, _frame, _classCost, _params, _max_curvature, x, z, heading, xp, zp, local_intermediate_heading, _safeZoneChecked, false);

                if (cost_graph_node_to_precomputed_node_with_connection_to_goal < 0)
                    continue;

                float total_cost = cost_graph_node_to_precomputed_node_with_connection_to_goal + get_cost_direct_connection_to_goal(_directConnection, W, xp, zp);

                long long lcost = static_cast<long long>(std::floor(100.0f * total_cost));
                // printf ("%d, %d direct connection to goal - %d, %d bestCost: %f cost: %f\n", x, z, xp, zp, bestCost, lcost);
                if (lcost <= _bestCost)
                {
                    // parent
                    _nodes[0].x = x;
                    _nodes[0].y = z;
                    _nodes[0].z = heading;
                    _nodes[0].w = cost_graph_node_to_precomputed_node_with_connection_to_goal;
                    // child
                    _nodes[1].x = xp;
                    _nodes[1].y = zp;
                    _nodes[1].z = local_intermediate_heading;
                    _nodes[1].w = total_cost;
                }
            }
        }
    }
};

bool CudaGraph::findBestGoalDirectConnection(float3 *og, float radius, bool isSafeZoneChecked)
{
    _bestNodeDirectConnection.get()[0].x = -1;
    _bestNodeDirectConnection.get()[0].y = -1;

    float max_curvature = _physicalParams.get()[PHYSICAL_MAX_CURVATURE];

    auto pcost = new CheckGoalReachedWithDirectConnectionCostProcess(
        _graph->getPtr(),
        _graphData->getPtr(),
        og,
        _graphGoalDirectConnection->getPtr(),
        _searchSpaceParams.get(),
        _classCosts.get(),
        radius,
        max_curvature,
        isSafeZoneChecked);

    pcost->runAndWait();
    _bestNodeDirectConnectionCost = pcost->bestCost();
    delete pcost;

    if (_bestNodeDirectConnectionCost >= std::numeric_limits<int>::max())
        return false;

    CheckGoalReachedWithDirectConnectionProcess(
        _graph->getPtr(),
        _graphData->getPtr(),
        og,
        _graphGoalDirectConnection->getPtr(),
        _searchSpaceParams.get(),
        _classCosts.get(),
        radius,
        max_curvature,
        isSafeZoneChecked,
        _bestNodeDirectConnectionCost,
        _bestNodeDirectConnection.get()).runAndWait();

    return true;

    // float best_cost = *cost.get() / 100;

    // return {(float)bestNode.get()->x, (float)bestNode.get()->y, best_cost};
}

float4 CudaGraph::bestGraphDirectConnectionParent()
{
    return {_bestNodeDirectConnection.get()[0].x, _bestNodeDirectConnection.get()[0].y, _bestNodeDirectConnection.get()[0].z, _bestNodeDirectConnection.get()[0].w};
}

float4 CudaGraph::bestGraphDirectConnectionChild()
{
    return {_bestNodeDirectConnection.get()[1].x, _bestNodeDirectConnection.get()[1].y, _bestNodeDirectConnection.get()[1].z, _bestNodeDirectConnection.get()[1].w};
}