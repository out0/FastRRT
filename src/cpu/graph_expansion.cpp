
#include "../../include/cuda_graph.h"
#include <fstream>

extern long computePos(int width, int x, int z);
extern float getHeadingCuda(float4 *graphData, long pos);
extern inline void setHeadingCuda(float4 *graphData, long pos, float heading);
extern void setTypeCuda(int4 *graph, long pos, int type);
extern int getTypeCuda(int4 *graph, long pos);
extern int2 getParentCuda(int4 *graph, long pos);
extern void setCostCuda(float4 *graphData, long pos, float cost);
extern float getCostCuda(float4 *graphData, long pos);
extern bool set(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);
extern bool setCollisionCuda(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost);
extern bool checkInGraphCuda(int4 *graph, long pos);
extern float generateRandom(RandState *state, int pos, float min_val, float max_val);
extern float generateRandomNeg(RandState *state, int pos, float max_val);
extern void setParentCuda(int4 *graph, long pos, int parent_x, int parent_z);
extern void incNodeDeriveCount(int4 *graph, long pos);
extern void setNodeDeriveCount(int4 *graph, long pos, int count);
extern int getNodeDeriveCount(int4 *graph, long pos);
extern bool canConnectToGoalUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_steering_rad, int x, int z, int goal_x, int goal_z, float goal_heading);
extern float getDirectCostCuda(float4 *graphData, long pos);
extern void setDirectCostCuda(float4 *graphData, long pos, float cost);
extern void assertDAGconsistency(int4 *graph, float4 *graphData, int width, int height, long pos);
extern void decNodeDeriveCount(int4 *graph, long pos);

extern bool preProcessedCollisionDistance(int *searchParams);
extern bool preProcessedCollisionVector(int *searchParams);
extern bool preProcessedDistanceToGoal(int *searchParams);

extern __device__ __host__ float4 expand_node(int4 *graph, float4 *graphData, float3 *frame, long pos, int x, int z, float steeringAngle_rad,
                                       float pathSize, float *classCosts, int *searchParams, double *physicalParams, float3 *ogCoordinateStart, float velocity_m_s, bool *nodeCollision,
                                       bool ignore_collision);

extern __device__ __host__ float4 checkDirectConnectionToGoal(float4 *graphData, float3 *frame,
                                                              float *classCosts, int *searchSpaceParams, float max_curvature,
                                                              int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading,
                                                              bool isSafeZoneChecked, bool isDistanceToGoalProcessed,
                                                              float distance_to_goal_tolerance,
                                                              float max_heading_error);

extern __device__ __host__ float4 expand_node(int4 *graph, float4 *graphData, float3 *frame, long pos, int x, int z, float steeringAngle_rad,
                                              float pathSize, float *classCosts, int *searchParams, double *physicalParams, float3 *ogCoordinateStart,
                                              float velocity_m_s, bool *nodeCollision, bool ignore_collision);

inline bool checkEquals(int2 &a, int2 &b)
{
    return a.x == b.x && a.y == b.y;
}

class AcceptDerivedNodesProcess : public ParallelProcessor
{

private:
    int4 *_graph;
    int _max;

public:
    AcceptDerivedNodesProcess(int4 *graph, int width,                                                                        //
                              int height, int numThreadHandlers = 12) :                                                      //
                                                                        ParallelProcessor(numThreadHandlers, height, width), //
                                                                        _graph(graph)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        if (getTypeCuda(_graph, pos) == GRAPH_TYPE_TEMP)
        {
            setTypeCuda(_graph, pos, GRAPH_TYPE_NODE);
        }
    }
};

void CudaGraph::acceptDerivedNodes(int2 goal, float goal_heading)
{
    AcceptDerivedNodesProcess(_graph->getPtr(),
                              _graph->width(),
                              _graph->height())
        .runAndWait();
}
void CudaGraph::acceptDerivedNode(int2 start, int2 lastNode)
{
    long pos = computePos(_graph->width(), lastNode.x, lastNode.y);
    setTypeCuda(_graph->getPtr(), pos, GRAPH_TYPE_NODE);
}


class NodeExpansionProcess : public ParallelProcessor
{

private:
    RandState *_state;
    int4 *_graph;
    float4 *_graphData;
    float3 *_frame;
    float *_classCosts;
    double *_physicalParams;
    int *_searchParams;
    float3 _ogCoordinateStart;
    float _maxPathSize;
    float _velocity_m_s;
    bool _controlExpansion;
    bool _forceExpansion;
    bool _nodeCollision;
    float _dist_to_goal_tolerance;
    float _heading_error_tolerance;
    int2 _goal;
    float _goal_heading;
    int _max;

public:
    NodeExpansionProcess(RandState *state, int4 *graph,                                                                                   //
                         float4 *graphData, float3 *frame,                                                                                //
                         float *classCosts, double *physicalParams,                                                                       //
                         int *searchParams, float3 ogCoordinateStart, float maxPathSize,                                                  //
                         float velocity_m_s, bool controlExpansion, bool forceExpansion,                                                  //
                         int2 goal, float goal_heading, float dist_to_goal_tolerance,                                                     //
                         float heading_error_tolerance, int numThreadHandlers = 12) :                                                     //
                                                                                      ParallelProcessor(numThreadHandlers,                //
                                                                                                        searchParams[FRAME_PARAM_HEIGHT], //
                                                                                                        searchParams[FRAME_PARAM_WIDTH]), //
                                                                                      _state(state),
                                                                                      _graph(graph),
                                                                                      _graphData(graphData),
                                                                                      _frame(frame),
                                                                                      _classCosts(classCosts),
                                                                                      _physicalParams(physicalParams),
                                                                                      _searchParams(searchParams),
                                                                                      _ogCoordinateStart(ogCoordinateStart),
                                                                                      _maxPathSize(maxPathSize),
                                                                                      _velocity_m_s(velocity_m_s),
                                                                                      _controlExpansion(controlExpansion),
                                                                                      _forceExpansion(forceExpansion),
                                                                                      _goal(goal),
                                                                                      _goal_heading(goal_heading),
                                                                                      _dist_to_goal_tolerance(dist_to_goal_tolerance),
                                                                                      _heading_error_tolerance(heading_error_tolerance)
    {
        _max = searchParams[FRAME_PARAM_HEIGHT] * searchParams[FRAME_PARAM_WIDTH];
        _nodeCollision = false;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        const int width = _searchParams[FRAME_PARAM_WIDTH];
        const int height = _searchParams[FRAME_PARAM_HEIGHT];

        if (pos >= width * height)
            return;

        if (!checkInGraphCuda(_graph, pos))
            return;

        if (_controlExpansion && getNodeDeriveCount(_graph, pos) > 0)
        {
            // printf("%d, %d has been derived too many times, skipping...\n", x, z);
            return;
        }

        int z = pos / width;
        int x = pos - z * width;

        float heading = getHeadingCuda(_graphData, pos);
        double maxSteeringAngle = _physicalParams[PHYSICAL_PARAM_MAX_STEERING_RAD];
        double maxCurvature = _physicalParams[PHYSICAL_PARAM_MAX_CURVATURE];

        //    printf ("max_curvature = %f\n", max_curvature);

        double steeringAngle = generateRandomNeg(_state, pos, maxSteeringAngle);
        double pathSize = 0;

        while (pathSize <= 0)
        {
            pathSize = generateRandom(_state, pos, 5.0, _maxPathSize);
        }

        /// during strong exponential expansion which we are not forcing graph expansion,
        // we can ignore graph collision to improve speed, because the graph is still expanding.
        // We did not reach a stuck state yet. If we keep colliding, we can make our graph reshape too early
        const bool ignore_collision = !_forceExpansion && !_controlExpansion;
        float4 result_node = expand_node(_graph, _graphData, _frame, pos, x, z, steeringAngle, pathSize,
                                         _classCosts, _searchParams, _physicalParams, &_ogCoordinateStart, _velocity_m_s, &_nodeCollision, ignore_collision);

        const bool node_expansion_successful = result_node.w == 1.0;

        if (node_expansion_successful)
        {
            const int x_new = TO_INT(result_node.x);
            const int z_new = TO_INT(result_node.y);
            const float heading_new = result_node.z;

            bool safeZoneChecked = preProcessedCollisionDistance(_searchParams);
            bool distToGoalChecked = preProcessedDistanceToGoal(_searchParams);

            float4 direct_connection = checkDirectConnectionToGoal(_graphData, _frame, _classCosts,
                                                                   _searchParams, maxCurvature, x_new, z_new, heading_new,
                                                                   _goal.x, _goal.y, _goal_heading, safeZoneChecked, distToGoalChecked,
                                                                   _dist_to_goal_tolerance, _heading_error_tolerance);
            const int last_x = TO_INT(direct_connection.x);
            const int last_z = TO_INT(direct_connection.y);
            const float last_heading = direct_connection.z;
            const float nodeCost = direct_connection.w;

            if (last_x < 0)
                return;

            const int direct_connection_pos = computePos(width, last_x, last_z);
            // creates (last_x, last_z) node in the graph (which is a final node candidate) and connects it tothe current new node directly
            set(_graph, _graphData, direct_connection_pos, last_heading, x_new, z_new, nodeCost, GRAPH_TYPE_TEMP, false);
        }
    }

    bool nodeCollision()
    {
        return _nodeCollision;
    }
};

void CudaGraph::expandTree(float3 *og, float maxPathSize, float velocity_m_s,
                           bool controlExpansion, bool forceExpansion, int2 goal, angle goal_heading,
                           float dist_to_goal_tolerance, angle heading_error_tolerance)
{
    auto p = new NodeExpansionProcess(_randState.get(),
                                      _graph->getPtr(),
                                      _graphData->getPtr(),
                                      og,
                                      _classCosts.get(),
                                      _physicalParams.get(),
                                      _searchSpaceParams.get(),
                                      _ogCoordinateStart,
                                      maxPathSize,
                                      velocity_m_s,
                                      controlExpansion,
                                      forceExpansion,
                                      goal,
                                      static_cast<float>(goal_heading.rad()),
                                      dist_to_goal_tolerance,
                                      static_cast<float>(heading_error_tolerance.rad()), 12);
    p->runAndWait();


    if (p->nodeCollision())
    {
        solveCollisions();
    }

    delete p;
}

float4 CudaGraph::derivateNode(float3 *og, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z)
{
    if (!checkInGraph(x, z))
        return float4{-1, -1, -1, 0};

    long pos = computePos(_graph->width(), x, z);

    bool nodeCollision = false;

    return expand_node(
        _graph->getPtr(),
        _graphData->getPtr(),
        og,
        pos, x, z,
        steeringAngle.rad(),
        pathSize,
        _classCosts.get(),
        _searchSpaceParams.get(),
        _physicalParams.get(),
        &_ogCoordinateStart,
        velocity_m_s,
        &nodeCollision,
        false);
}

bool CudaGraph::canConnectToGoal(SearchFrame *search_frame, int x, int z, int goal_x, int goal_z, int goal_heading)
{
    if (search_frame->isObstacle(goal_x, goal_z))
        return false;

    float maxSteering = _physicalParams.get()[PHYSICAL_PARAM_MAX_STEERING_RAD];

    return canConnectToGoalUsingHermite(
        _graph->getPtr(),
        _graphData->getPtr(),
        search_frame->getPtr(),
        search_frame->getClassCostsPtr(),
        search_frame->getFrameParamsPtr(),
        maxSteering,
        x, z, goal_x, goal_z, goal_heading);
}

void CudaGraph::dumpNodesToFile(const char *filename)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open())
        return;
    std::vector<int3> nodes = listAll();

    for (int3 n : nodes)
    {
        GraphNode g(n.x, n.y, n.z);
        int2 parent = getParent(n.x, n.y);
        int parent_x = parent.x;
        int parent_z = parent.y;
        float heading_rad = getHeading(n.x, n.y).rad();
        float cost = getCost(n.x, n.y);
        float connectToEndCost = getDirectCost(n.x, n.z);
        ofs << n.x << " "
            << n.y << " "
            << heading_rad << " "
            << n.z << " "
            << parent_x << " "
            << parent_z << " "
            << connectToEndCost << " "
            << cost << "\n";
    }

    ofs.close();
}
