
#include <driveless/cuda_basic.h>
#include "../../include/cuda_graph.h"
#include <driveless/cpu_parallel_processor.h>
#include "atomic_utils.h"

extern bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern float getCostCpu(float4 *graphData, long pos);
extern long computePos(int width, int x, int z);
extern float getHeadingCpu(float4 *graphData, long pos);

#define K1 1
#define K2 3
#define K3 1

__device__ long long __compute_cost_findBestNode(float dist, float heading_rad, float nodeCost)
{
    return static_cast<long long>(K1 * dist + K2 * TO_DEG * heading_rad + K3 * nodeCost);
}

class FindBestNodeWithHeadingBestCostProcess : public ParallelProcessor
{
private:
    int4 *_graph;
    float4 *_graphData;
    float3 *_frame;
    int *_params;
    float *_classCost;
    float _searchRadius;
    int _targetX;
    int _targetZ;
    float _targetHeading_rad;
    float _maxHeadingError_rad;
    int _max;
    std::atomic<long long> _bestCost;

public:
    FindBestNodeWithHeadingBestCostProcess(int4 *graph,
                                           float4 *graphData,
                                           float3 *frame,
                                           int *params,
                                           float *classCost,
                                           float searchRadius,
                                           int targetX,
                                           int targetZ,
                                           float targetHeading_rad,
                                           float maxHeadingError_rad, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]),
                                                                                                    _graph(graph), _graphData(graphData),                     //
                                                                                                    _frame(frame), _params(params), _classCost(classCost),    //
                                                                                                    _searchRadius(searchRadius), _targetX(targetX),           //
                                                                                                    _targetZ(targetZ), _targetHeading_rad(targetHeading_rad), //
                                                                                                    _maxHeadingError_rad(maxHeadingError_rad)
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
        int minDistX = _params[FRAME_PARAM_MIN_DIST_X];
        int minDistZ = _params[FRAME_PARAM_MIN_DIST_Z];

        int z = pos / width;
        int x = pos - z * width;

        if (_graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
            return;

        int dx = _targetX - x;
        int dz = _targetZ - z;

        const float dist = sqrtf(dx * dx + dz * dz);

        if (dist > _searchRadius)
        {
            // printf ("%d, %d failed because of dist\n", x, z);
            return;
        }

        float heading = getHeadingCpu(_graphData, pos);

        if (abs(heading - _targetHeading_rad) > _maxHeadingError_rad)
        {
            // printf ("%d, %d failed because of heading error: %f vs %f\n", x, z, abs(heading - targetHeading_rad), maxHeadingError_rad);
            return;
        }

        if (!__computeFeasibleForAngle(_frame, _params, _classCost, minDistX, minDistZ, x, z, heading))
        {
            // printf ("%d, %d, %f failed because is unfeasible minDistX = %d, minDistZ = %d\n", x, z, heading, minDistX, minDistZ);
            return;
        }

        long long cost = __compute_cost_findBestNode(dist, heading, getCostCpu(_graphData, pos));

        atomicMin(_bestCost, cost);
    }

    long long bestCost()
    {
        return _bestCost;
    }
};

class FindBestNodeWithHeadingFirstNodeWithCostProcess : public ParallelProcessor
{
private:
    int4 *_graph;
    float4 *_graphData;
    float3 *_frame;
    int *_params;
    float *_classCost;
    float _searchRadius;
    int _targetX;
    int _targetZ;
    float _targetHeading_rad;
    float _maxHeadingError_rad;
    int _max;
    int2 _node;
    long long _bestCost;

public:
    FindBestNodeWithHeadingFirstNodeWithCostProcess(int4 *graph,
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
                                                    int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]),
                                                                                  _graph(graph), _graphData(graphData),                     //
                                                                                  _frame(frame), _params(params), _classCost(classCost),    //
                                                                                  _searchRadius(searchRadius), _targetX(targetX),           //
                                                                                  _targetZ(targetZ), _targetHeading_rad(targetHeading_rad), //
                                                                                  _maxHeadingError_rad(maxHeadingError_rad), _bestCost(bestCost)
    {
        _max = params[FRAME_PARAM_HEIGHT] * params[FRAME_PARAM_WIDTH];
        _node = {-1, -1};
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int width = _params[FRAME_PARAM_WIDTH];
        int height = _params[FRAME_PARAM_HEIGHT];
        int minDistX = _params[FRAME_PARAM_MIN_DIST_X];
        int minDistZ = _params[FRAME_PARAM_MIN_DIST_Z];

        int z = pos / width;
        int x = pos - z * width;

        if (_graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
            return;

        int dx = _targetX - x;
        int dz = _targetZ - z;

        const float dist = sqrtf(dx * dx + dz * dz);

        if (dist > _searchRadius)
            return;

        float heading = getHeadingCpu(_graphData, pos);

        if (abs(heading - _targetHeading_rad) > _maxHeadingError_rad)
            return;

        if (!__computeFeasibleForAngle(_frame, _params, _classCost, minDistX, minDistZ, x, z, heading))
        {
            return;
        }

        long long cost = __compute_cost_findBestNode(dist, heading, getCostCpu(_graphData, pos));

        if (cost == _bestCost)
        {
            _node.x = x;
            _node.y = z;
        }
    }

    int2 bestNode()
    {
        return _node;
    }
};

long long CudaGraph::findBestNodeCost(float3 *og, angle heading, float radius, int x, int z, float maxHeadingError) {
    auto p1 = new FindBestNodeWithHeadingBestCostProcess(_graph->getPtr(),
                                                                  _graphData->getPtr(),
                                                                  og,
                                                                  _searchSpaceParams.get(),
                                                                  _classCosts.get(),
                                                                  radius,
                                                                  x, z,
                                                                  heading.rad(),
                                                                  maxHeadingError);

    p1->runAndWait();

    long long  res = p1->bestCost();

    delete p1;
    return res;
}

int2 CudaGraph::findBestNode(float3 *og, angle heading, float radius, int x, int z, float maxHeadingError, long long cost)
{
    auto p2 = new FindBestNodeWithHeadingFirstNodeWithCostProcess(_graph->getPtr(),
                                                                  _graphData->getPtr(),
                                                                  og,
                                                                  _searchSpaceParams.get(),
                                                                  _classCosts.get(),
                                                                  radius,
                                                                  x, z,
                                                                  heading.rad(),
                                                                  maxHeadingError,
                                                                  cost);

    p2->runAndWait();

    int2 res = p2->bestNode();

    delete p2;
    return res;
}

extern double compute_euclidean_2d_dist(const int2 &start, const int2 &end);

class CheckGoalReachedProcess : public ParallelProcessor
{
    int4 *_graph;
    float4 *_graphData;
    float3 *_frame;
    int *_params;
    float *_costs;
    int _goalX;
    int _goalZ;
    float _goalHeading;
    float _distToGoalTolerance;
    float _maxHeadingError;
    bool _goalReached;
    int _max;

public:
    CheckGoalReachedProcess(int4 *graph,
                            float4 *graphData,
                            float3 *frame,
                            int *params,
                            float *classCost,
                            int goalX,
                            int goalZ,
                            float goalHeading,
                            float distToGoalTolerance,
                            float maxHeadingError,
                            int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]),
                                                          _graph(graph), _graphData(graphData),                                 //
                                                          _frame(frame), _params(params),                                       //
                                                          _goalX(goalX), _goalZ(goalZ),                                         //
                                                          _goalHeading(goalHeading), _distToGoalTolerance(distToGoalTolerance), //
                                                          _maxHeadingError(maxHeadingError)
    {
        _max = params[FRAME_PARAM_HEIGHT] * params[FRAME_PARAM_WIDTH];
        _goalReached = false;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int width = _params[FRAME_PARAM_WIDTH];
        int height = _params[FRAME_PARAM_HEIGHT];

        if (_graph[pos].z != GRAPH_TYPE_NODE)
            return;

        int z = pos / width;
        int x = pos - z * width;

        int2 s = {x, z};
        int2 e = {_goalX, +_goalZ};

        if (compute_euclidean_2d_dist(s, e) > _distToGoalTolerance)
            return;

        float heading = getHeadingCpu(_graphData, pos);

        if (abs(heading - _goalHeading) <= _maxHeadingError)
            _goalReached = true;
    }

    bool goalReached()
    {
        return _goalReached;
    }
};

bool CudaGraph::checkGoalReached(float3 *og, int2 goal, angle heading, float distanceToGoalTolerance, float maxHeadingError)
{

    auto p = new CheckGoalReachedProcess(_graph->getPtr(),
                                         _graphData->getPtr(),
                                         og,
                                         _searchSpaceParams.get(),
                                         _classCosts.get(),
                                         goal.x,
                                         goal.y,
                                         (float)heading.rad(),
                                         distanceToGoalTolerance,
                                         maxHeadingError);



                                         
    p->runAndWait();
    bool res = p->goalReached();
    delete p;
    return res;
}