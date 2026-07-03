
#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include "../../include/cuda_graph.h"

extern float4 check_kinematic_new_path(int4 *graph, float4 *graphData, double *physicalParams, int *searchSpaceParams, float3 *frame, float *classCosts, float3 *ogStart, int2 start, float steeringAngle, float pathSize, float velocity_m_s);
extern bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern long computePos(int width, int x, int z);
extern float getHeadingCpu(float4 *graphData, long pos);
extern void setTypeCpu(int4 *graph, long pos, int type);
extern int getTypeCpu(int4 *graph, long pos);
extern int2 getParentCpu(int4 *graph, long pos);
extern void setCostCpu(float4 *graphData, long pos, float cost);
extern float getCostCpu(float4 *graphData, long pos);
extern bool set(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);
extern bool setCollisionCpu(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost);
extern bool checkInGraphCpu(int4 *graph, long pos);
extern float generateRandom(RandState *state, int pos, float min_val, float max_val);
extern float generateRandomNeg(RandState *state, int pos, float max_val);
extern void setParentCpu(int4 *graph, long pos, int parent_x, int parent_z);
extern void incNodeDeriveCount(int4 *graph, long pos);
extern void decNodeDeriveCount(int4 *graph, long pos);
extern int getNodeDeriveCount(int4 *graph, long pos);
extern void setNodeDeriveCount(int4 *graph, long pos, int count);
extern float canConnectToGoalUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_steering_rad, int x, int z, int goal_x, int goal_z, float goal_heading);
extern void setDirectCostCpu(float4 *graphData, long pos, float cost);
extern void assertDAGconsistency(int4 *graph, float4 *graphData, int width, int height, long pos);
extern int2 expand_node(int4 *graph, float4 *graphData, float3 *frame, long pos, int x, int z, float steeringAngle_rad,
                        float pathSize, float *classCosts, int *searchParams, double *physicalParams, float3 ogStart, float velocity_m_s, bool *nodeCollision);

#define MIN_PATH_SIZE 5.0

#define BLOCK_SIZE 128
#define CHECK_NO_COLLISION 1

int computeDensityPos(int density_width, int x, int z)
{
    int density_x = TO_INT(x / BLOCK_SIZE);
    int density_z = TO_INT(z / BLOCK_SIZE);
    return (density_z * density_width + density_x);
}

class CountNodesInDensityRegionProcess : public ParallelProcessor
{

private:
    int4 *_graph;
    int *_params;
    unsigned int *_node_count;
    int _max;

public:
    CountNodesInDensityRegionProcess(int4 *graph, int *params,                                                                                                  //
                                     unsigned int *node_count,                                                                                                  //
                                     int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]), //
                                                                   _graph(graph),
                                                                   _params(params), //
                                                                   _node_count(node_count)
    {
        _max = params[FRAME_PARAM_WIDTH] * params[FRAME_PARAM_HEIGHT];
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int width = _params[FRAME_PARAM_WIDTH];
        int height = _params[FRAME_PARAM_HEIGHT];
        int density_width = _params[FRAME_DENSITY_WIDTH];

        int type = getTypeCpu(_graph, pos);

        if (type == GRAPH_TYPE_NULL || type == GRAPH_TYPE_PROCESSING)
            return;

        int z = pos / width;
        int x = pos - z * width;

        const int densityPos = computeDensityPos(density_width, x, z);

        // printf ("%d, %d incrementing density region %d\n", x, z, densityPos);
        __atomic_fetch_add(&_node_count[densityPos], 1, __ATOMIC_SEQ_CST);
    }
};

bool checkCanExpand(int4 *graph, unsigned int *region_count, int *params, float node_mean, int pos, int x, int z, bool expandFrontier)
{
    if (expandFrontier)
    {
        return getNodeDeriveCount(graph, pos) == 0;
    }

    const int densityPos = computeDensityPos(params[FRAME_DENSITY_WIDTH], x, z);
    // return getNodeDeriveCount(graph, pos) < 3 && (region_count[densityPos] <= 0.5 * BLOCK_SIZE);
    return region_count[densityPos] <= 0.5 * BLOCK_SIZE;
}

void CudaGraph::__initializeRegionDensity()
{
    int width = _searchSpaceParams.get()[FRAME_PARAM_WIDTH];
    int height = _searchSpaceParams.get()[FRAME_PARAM_HEIGHT];

    int density_width = TO_INT(width / BLOCK_SIZE) + 1;
    int density_height = TO_INT(height / BLOCK_SIZE) + 1;
    int density_size = density_width * density_height;

    _searchSpaceParams.get()[FRAME_DENSITY_WIDTH] = density_width;
    _searchSpaceParams.get()[FRAME_DENSITY_HEIGHT] = density_height;
    _searchSpaceParams.get()[FRAME_DENSITY_SIZE] = density_size;

    // printf("graph size: %d, %d\n", width, height);
    // printf("num of density regions: %d\n", density_size);
    // printf("density region size: %d x %d\n", density_width, density_height);

    _region_node_count = std::make_unique<unsigned int[]>(density_size);
    for (int i = 0; i < density_size; i++)
    {
        _region_node_count.get()[i] = 0;
    }

    _node_mean = 0;
}

void CudaGraph::__dealocRegionDensity()
{
    _region_node_count = nullptr;
}

void CudaGraph::computeGraphRegionDensity()
{
    int density_size = _searchSpaceParams.get()[FRAME_DENSITY_SIZE];

    for (int i = 0; i < density_size; i++)
    {
        _region_node_count.get()[i] = 0;
    }

    CountNodesInDensityRegionProcess(_graph->getPtr(),
                                     _searchSpaceParams.get(),
                                     _region_node_count.get())
        .runAndWait();

    _node_mean = 0;
    int numRegionsWithNodes = 0;
    for (int i = 0; i < density_size; i++)
    {
        _node_mean += _region_node_count.get()[i];
        if (_region_node_count.get()[i] > 0)
        {
            numRegionsWithNodes++;
            // printf("(+) region %i: %d\n", i, _region_node_count[i]);
        }
    }

    if (numRegionsWithNodes == 0)
    {
        _node_mean = 0;
        return;
    }

    // printf("region total: %d\n", _node_mean);
    _node_mean = TO_INT(_node_mean / numRegionsWithNodes);
    // printf("region mean: %d\n", _node_mean);

    for (int i = 0; i < density_size; i++)
    {
        if (_region_node_count.get()[i] == 0)
            continue;
        int density_x = TO_INT(i % _searchSpaceParams.get()[FRAME_DENSITY_WIDTH]);
        int density_z = TO_INT(i / _searchSpaceParams.get()[FRAME_DENSITY_WIDTH]);
        // printf("density region (%d, %d): %d\n", density_x, density_z, _region_node_count[i]);
    }
}

class SmartNodeExpansionProcessor : public ParallelProcessor
{

private:
    RandState *_state;
    int4 *_graph;
    float4 *_graphData;
    float3 *_frame;
    unsigned int *_region_count;
    int _node_mean;
    float *_classCosts;
    int *_searchParams;
    double *_physicalParams;
    float3 _ogStart;
    float _maxPathSize;
    float _velocity_m_s;
    bool _expandFrontier;
    bool _forceExpand;
    bool *_nodeCollision;
    int _goal_x;
    int _goal_z;
    float _goal_heading;
    int _max;

public:
    SmartNodeExpansionProcessor(RandState *state, int4 *graph,
                                float4 *graphData, float3 *frame,            //
                                unsigned int *region_count, int node_mean,   //
                                float *classCosts, int *searchParams,        //
                                double *physicalParams, float3 ogStart,      //
                                float maxPathSize, float velocity_m_s,       //
                                bool expandFrontier, bool forceExpand,       //
                                bool *nodeCollision, int goal_x, int goal_z, //
                                float goal_heading, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, searchParams[FRAME_PARAM_HEIGHT], searchParams[FRAME_PARAM_WIDTH]),
                                                                                  _state(state),
                                                                                  _graph(graph),
                                                                                  _graphData(graphData),
                                                                                  _frame(frame),
                                                                                  _region_count(region_count),
                                                                                  _node_mean(node_mean),
                                                                                  _classCosts(classCosts),
                                                                                  _searchParams(searchParams),
                                                                                  _physicalParams(physicalParams),
                                                                                  _ogStart(ogStart),
                                                                                  _maxPathSize(maxPathSize),
                                                                                  _velocity_m_s(velocity_m_s),
                                                                                  _expandFrontier(expandFrontier),
                                                                                  _forceExpand(forceExpand),
                                                                                  _nodeCollision(nodeCollision),
                                                                                  _goal_x(goal_x),
                                                                                  _goal_z(goal_z),
                                                                                  _goal_heading(goal_heading)
    {
        _max = searchParams[FRAME_PARAM_WIDTH] * searchParams[FRAME_PARAM_HEIGHT];
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int width = _searchParams[FRAME_PARAM_WIDTH];
        int height = _searchParams[FRAME_PARAM_HEIGHT];

        if (!checkInGraphCpu(_graph, pos))
            return;

        int z = pos / width;
        int x = pos - z * width;

        // Smart expansion: if this node is a leaf, it can be expanded. If not, it still can be expanded if the region density is lower than the mean density
        if (!_forceExpand && !checkCanExpand(_graph, _region_count, _searchParams, _node_mean, pos, x, z, _expandFrontier))
        {
            // const int densityPos = computeDensityPos(searchParams[FRAME_DENSITY_WIDTH], x, z);
            // printf("wont expand (%d, %d) because of density: %d vs mean %d\n", x, z, region_count[densityPos], node_mean);
            return;
        }

        float heading = getHeadingCpu(_graphData, pos);
        double maxSteeringAngle = _physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

        double steeringAngle = generateRandomNeg(_state, pos, maxSteeringAngle);
        double pathSize = generateRandom(_state, pos, 5.0, _maxPathSize);
        if (pathSize <= 0)
            pathSize = MIN_PATH_SIZE;

        expand_node(_graph, _graphData, _frame, pos, x, z, steeringAngle, pathSize, _classCosts, _searchParams, _physicalParams, _ogStart, _velocity_m_s, _nodeCollision);
    }
};

void CudaGraph::smartExpansion(float3 *og, float maxPathSize, float velocity_m_s, bool expandFrontier, bool forceExpand, int2 goal, angle goal_heading, float dist_to_goal_tolerance, angle heading_error_tolerance)
{
    bool nodeCollision = false;

    SmartNodeExpansionProcessor(
        _randState.get(),
        _graph->getPtr(),
        _graphData->getPtr(),
        og,
        _region_node_count.get(),
        _node_mean,
        _classCosts.get(),
        _searchSpaceParams.get(),
        _physicalParams.get(),
        _ogCoordinateStart,
        maxPathSize,
        velocity_m_s,
        expandFrontier,
        forceExpand,
        &nodeCollision,
        goal.x,
        goal.y,
        goal_heading.rad())
        .runAndWait();

    computeGraphRegionDensity();

    if (nodeCollision)
    {
        // printf("solving graph collision\n");
        solveCollisions();
    }
}