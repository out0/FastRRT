
#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include "../../include/cuda_graph.h"
#include <fstream>

extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ inline void setHeadingCuda(float4 *graphData, long pos, float heading);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);
extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);
extern __device__ __host__ int2 getParentCuda(int4 *graph, long pos);
extern __device__ __host__ void setCostCuda(float4 *graphData, long pos, float cost);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ bool set(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);
extern __device__ __host__ bool setCollisionCuda(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost);
extern __device__ __host__ bool checkInGraphCuda(int4 *graph, long pos);
extern __device__ float generateRandom(curandState *state, int pos, float min_val, float max_val);
extern __device__ float generateRandomNeg(curandState *state, int pos, float max_val);
extern __device__ __host__ void setParentCuda(int4 *graph, long pos, int parent_x, int parent_z);
extern __device__ __host__ void incNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ void setNodeDeriveCount(int4 *graph, long pos, int count);
extern __device__ __host__ int getNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ bool canConnectToGoalUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_steering_rad, int x, int z, int goal_x, int goal_z, float goal_heading);
extern __device__ __host__ float getDirectCostCuda(float4 *graphData, long pos);
extern __device__ __host__ void setDirectCostCuda(float4 *graphData, long pos, float cost);
extern __device__ __host__ void assertDAGconsistency(int4 *graph, float4 *graphData, int width, int height, long pos);
extern __device__ __host__ void decNodeDeriveCount(int4 *graph, long pos);

extern __device__ __host__ float4 checkDirectConnectionToGoal(float4 *graphData, float3 *frame,
                                                              float *classCosts, int *searchSpaceParams, float max_curvature,
                                                              int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading,
                                                              bool isSafeZoneChecked, bool isDistanceToGoalProcessed,
                                                              float distance_to_goal_tolerance,
                                                              float max_heading_error);

extern __device__ __host__ bool preProcessedCollisionDistance(int *searchParams);
extern __device__ __host__ bool preProcessedCollisionVector(int *searchParams);
extern __device__ __host__ bool preProcessedDistanceToGoal(int *searchParams);
extern __device__ __host__ float4 expand_node(int4 *graph, float4 *graphData, float3 *frame, long pos, int x, int z, float steeringAngle_rad,
                                              float pathSize, float *classCosts, int *searchParams, double *physicalParams, float3 *ogCoordinateStart,
                                              float velocity_m_s, bool *nodeCollision, bool ignore_collision);

__device__ __host__ inline bool checkEquals(int2 &a, int2 &b)
{
    return a.x == b.x && a.y == b.y;
}

__global__ void __CUDA_accept_derived_nodes(int4 *graph, float4 *graphData, int goal_x, int goal_z, float goal_heading, bool *goalReached, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (getTypeCuda(graph, pos) == GRAPH_TYPE_TEMP)
    {
        setTypeCuda(graph, pos, GRAPH_TYPE_NODE);
    }
    // else if (getTypeCuda(graph, pos) == GRAPH_TYPE_CONNECT_TO_GOAL)
    // {
    //     int z = pos / width;
    //     int x = pos - z * width;

    //     float currentDirectCost = getDirectCostCuda(graphData, pos);
    //     if (*bestCostDirectConnect >= TO_INT(1000 * currentDirectCost))
    //     {
    //         printf("found the best node %d, %d to connect to the goal: %d, %d with cost %f\n", x, z, goal_x, goal_z, currentDirectCost);
    //         long goalPos = computePos(width, goal_x, goal_z);
    //         float parentCost = getCostCuda(graphData, pos);
    //         set(graph, graphData, goalPos, goal_heading, x, z, parentCost + currentDirectCost, GRAPH_TYPE_NODE, true);
    //         *goalReached = true;
    //     }
    //     setTypeCuda(graph, pos, GRAPH_TYPE_NODE);
    // }

    // atomicCAS(&(graph[pos].z), GRAPH_TYPE_TEMP, GRAPH_TYPE_NODE);
}
void CudaGraph::acceptDerivedNodes(int2 goal, float goal_heading)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_accept_derived_nodes<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getPtr(),
        _graphData->getPtr(),
        goal.x,
        goal.y,
        goal_heading,
        _goalReached->get(),
        _graph->width(),
        _graph->height());

    CUDA(cudaDeviceSynchronize());
}
void CudaGraph::acceptDerivedNode(int2 start, int2 lastNode)
{
    long pos = computePos(_graph->width(), lastNode.x, lastNode.y);
    setTypeCuda(_graph->getPtr(), pos, GRAPH_TYPE_NODE);
}

__global__ void __CUDA_random_node_expansion(curandState *state, int4 *graph, float4 *graphData,
                                             float3 *frame, float *classCosts, double *physicalParams, int *searchParams, float3 *ogCoordinateStart,
                                             float maxPathSize, float velocity_m_s, bool controlExpansion, bool forceExpansion, bool *nodeCollision,
                                             int2 goal, float goal_heading, float distToGoalTolerance, float maxHeadingError)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = searchParams[FRAME_PARAM_WIDTH];
    const int height = searchParams[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (!checkInGraphCuda(graph, pos))
        return;

    if (controlExpansion && getNodeDeriveCount(graph, pos) > 0)
    {
        // printf("%d, %d has been derived too many times, skipping...\n", x, z);
        return;
    }

    int z = pos / width;
    int x = pos - z * width;

    float heading = getHeadingCuda(graphData, pos);
    double maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];
    double maxCurvature = physicalParams[PHYSICAL_MAX_CURVATURE];

    //    printf ("max_curvature = %f\n", max_curvature);

    double steeringAngle = generateRandomNeg(state, pos, maxSteeringAngle);
    double pathSize = 0;

    while (pathSize <= 0)
    {
        pathSize = generateRandom(state, pos, 5.0, maxPathSize);
    }

    /// during strong exponential expansion which we are not forcing graph expansion,
    // we can ignore graph collision to improve speed, because the graph is still expanding.
    // We did not reach a stuck state yet. If we keep colliding, we can make our graph reshape too early
    const bool ignore_collision = !forceExpansion && !controlExpansion;

    float4 result_node = expand_node(graph, graphData, frame, pos, x, z, steeringAngle, pathSize, classCosts, searchParams, physicalParams, ogCoordinateStart, velocity_m_s, nodeCollision, ignore_collision);

    const bool node_expansion_successful = result_node.w == 1.0;

    if (node_expansion_successful)
    {
        const int x_new = TO_INT(result_node.x);
        const int z_new = TO_INT(result_node.y);
        const float heading_new = result_node.z;

        bool safeZoneChecked = preProcessedCollisionDistance(searchParams);
        bool distToGoalChecked = preProcessedDistanceToGoal(searchParams);

        float4 direct_connection = checkDirectConnectionToGoal(graphData, frame, classCosts,
                                                               searchParams, maxCurvature, x_new, z_new, heading_new,
                                                               goal.x, goal.y, goal_heading, safeZoneChecked, distToGoalChecked,
                                                               distToGoalTolerance, maxHeadingError);
        const int last_x = TO_INT(direct_connection.x);
        const int last_z = TO_INT(direct_connection.y);
        const float last_heading = direct_connection.z;
        const float nodeCost = direct_connection.w;

        if (last_x < 0)
            return;

        const int direct_connection_pos = computePos(width, last_x, last_z);
        // creates (last_x, last_z) node in the graph (which is a final node candidate) and connects it tothe current new node directly
        set(graph, graphData, direct_connection_pos, last_heading, x_new, z_new, nodeCost, GRAPH_TYPE_TEMP, false);
    }
}

void CudaGraph::expandTree(float3 *og, float maxPathSize, float velocity_m_s,
                           bool controlExpansion, bool forceExpansion, int2 goal, angle goal_heading,
                           float dist_to_goal_tolerance, angle heading_error_tolerance)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_nodeCollision->get() = false;

    __CUDA_random_node_expansion<<<numBlocks, THREADS_IN_BLOCK>>>(
        _randState->get(),
        _graph->getPtr(),
        _graphData->getPtr(),
        og,
        _classCosts->get(),
        _physicalParams->get(),
        _searchSpaceParams->get(),
        _ogCoordinateStart->get(),
        maxPathSize,
        velocity_m_s,
        controlExpansion,
        forceExpansion,
        _nodeCollision->get(),
        goal,
        goal_heading.rad(),
        dist_to_goal_tolerance,
        heading_error_tolerance.rad());

    CUDA(cudaDeviceSynchronize());

    // dumpNodesToFile("before_collision.txt");

    if (*_nodeCollision->get())
    {
        // printf("Collision detected, solving...\n");

        solveCollisions();
        // dumpNodesToFile("after_collision.txt");
    }
}

float4 CudaGraph::derivateNode(float3 *og, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z)
{
    if (!checkInGraph(x, z))
        return float4{-1, -1, -1, 0};

    long pos = computePos(_graph->width(), x, z);

    return expand_node(
        _graph->getPtr(),
        _graphData->getPtr(),
        og,
        pos, x, z,
        steeringAngle.rad(),
        pathSize,
        _classCosts->get(),
        _searchSpaceParams->get(),
        _physicalParams->get(),
        _ogCoordinateStart->get(),
        velocity_m_s,
        _nodeCollision->get(),
        false);
}

bool CudaGraph::canConnectToGoal(SearchFrame *search_frame, int x, int z, int goal_x, int goal_z, int goal_heading)
{
    if (search_frame->isObstacle(goal_x, goal_z))
        return false;

    float maxSteering = _physicalParams->get()[PHYSICAL_PARAMS_MAX_STEERING_RAD];

    return canConnectToGoalUsingHermite(
        _graph->getPtr(),
        _graphData->getPtr(),
        search_frame->getPtr(),
        search_frame->getCudaClassCostsPtr(),
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
