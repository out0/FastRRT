
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../../include/graph.h"
#include <fstream>

extern __device__ __host__ float4 check_kinematic_new_path(int4 *graph, float4 *graphData, double *physicalParams, int *searchSpaceParams, float3 *frame, float *classCosts, float3 *ogStart, int2 start, float steeringAngle, float pathSize, float velocity_m_s);
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
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
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
    setTypeCuda(_graph->getCudaPtr(), pos, GRAPH_TYPE_NODE);
}


__device__ __host__ bool change_graph_type_if_current_value_equals_expected_value(int4 *graph, long pos, int expected_value, int new_value) {

#ifdef __CUDA_ARCH__
    return atomicCAS(&(graph[pos].z), expected_value, new_value) == expected_value;
#else
    if (graph[pos].z == expected_value) {
        graph[pos].z = new_value;
        return true;
    }
    return false;
#endif
}

__device__ __host__ int2 expand_node(int4 *graph, float4 *graphData, float3 *frame, long pos, int x, int z, float steeringAngle_rad, 
    float pathSize, float *classCosts, int *searchParams, double *physicalParams, float3 *ogStart, float velocity_m_s, bool *nodeCollision)
{
    int width = searchParams[FRAME_PARAM_WIDTH];

    float heading = getHeadingCuda(graphData, pos);

    float4 end = check_kinematic_new_path(graph, graphData, physicalParams, searchParams, frame, classCosts, ogStart, {x, z}, steeringAngle_rad, pathSize, velocity_m_s);

    // printf("end expansion: %f, %f, heading: %f, cost: %f\n", end.x, end.y, end.w, end.z);

    if (end.x < 0 || end.y < 0)
        return {-1, -1};

    int end_x = TO_INT(end.x);
    int end_z = TO_INT(end.y);

    if (end_x == ogStart->x && end_z == ogStart->y) {
        return {-1, -1};
    }

    float end_cost = end.z;
    float end_heading = end.w;

    long end_pos = computePos(width, end_x, end_z);

    if (end_pos == pos)
        return {-1, -1};

    if (change_graph_type_if_current_value_equals_expected_value(graph, end_pos, GRAPH_TYPE_NULL, GRAPH_TYPE_TEMP)) {
        // A new node is being added to the graph
        incNodeDeriveCount(graph, pos);
        set(graph, graphData, end_pos, end_heading, x, z, end_cost, GRAPH_TYPE_TEMP, true);
    }
    
    if (change_graph_type_if_current_value_equals_expected_value(graph, end_pos, GRAPH_TYPE_NODE, GRAPH_TYPE_COLLISION)) {
        set(graph, graphData, end_pos, end_heading, x, z, end_cost, GRAPH_TYPE_COLLISION, true);
        *nodeCollision = true;
        decNodeDeriveCount(graph, pos);
    }

    return {TO_INT(end.x), TO_INT(end.y)};
}


__global__ void __CUDA_random_node_expansion(curandState *state, int4 *graph, float4 *graphData, float3 *frame, float *classCosts, double *physicalParams, int *searchParams, float3 *ogStart, float maxPathSize, float velocity_m_s, bool frontierExploration, bool *nodeCollision, long start_node_pos, int2 goal, float goal_heading)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = searchParams[FRAME_PARAM_WIDTH];
    const int height = searchParams[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (!checkInGraphCuda(graph, pos))
        return;

    if (frontierExploration && getNodeDeriveCount(graph, pos) > 0)
    {
        // printf("%d, %d has been derived too many times, skipping...\n", x, z);
        return;
    }

    int z = pos / width;
    int x = pos - z * width;

    float heading = getHeadingCuda(graphData, pos);
    double maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

//    printf ("max_curvature = %f\n", max_curvature);

    double steeringAngle = generateRandomNeg(state, pos, maxSteeringAngle);
    double pathSize = 0;

    while (pathSize <= 0)
    {
        pathSize = generateRandom(state, pos, 5.0, maxPathSize);
    }

    expand_node(graph, graphData, frame, pos, x, z, steeringAngle, pathSize, classCosts, searchParams, physicalParams, ogStart, velocity_m_s, nodeCollision);
}


void CudaGraph::expandTree(float3 *og, angle goalHeading, float maxPathSize, float velocity_m_s, bool frontierExpansion, int2 start_node, int2 goal, angle goal_heading)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_nodeCollision->get() = false;
    const long start_node_pos = computePos(_graph->width(), start_node.x, start_node.y);

    __CUDA_random_node_expansion<<<numBlocks, THREADS_IN_BLOCK>>>(
        _randState->get(),
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _classCosts->get(),
        _physicalParams->get(),
        _searchSpaceParams->get(),
        _ogCoordinateStart->get(),
        maxPathSize,
        velocity_m_s,
        frontierExpansion,
        _nodeCollision->get(),
        start_node_pos,
        goal,
        goal_heading.rad());

    CUDA(cudaDeviceSynchronize());

    //dumpNodesToFile("before_collision.txt");

    if (*_nodeCollision->get())
    {
       // printf("Collision detected, solving...\n");

        solveCollisions();
       // dumpNodesToFile("after_collision.txt");
    }
}

int2 CudaGraph::derivateNode(float3 *og, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z)
{
    if (!checkInGraph(x, z))
        return int2{-1, -1};

    long pos = computePos(_graph->width(), x, z);

    return expand_node(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        pos, x, z,
        steeringAngle.rad(),
        pathSize,
        _classCosts->get(),
        _searchSpaceParams->get(),
        _physicalParams->get(),
        _ogCoordinateStart->get(),
        velocity_m_s,
        _nodeCollision->get());
}

bool CudaGraph::canConnectToGoal(SearchFrame *search_frame, int x, int z, int goal_x, int goal_z, int goal_heading)
{
    if (search_frame->isObstacle(goal_x, goal_z))
        return false;

    float maxSteering = _physicalParams->get()[PHYSICAL_PARAMS_MAX_STEERING_RAD];

    return canConnectToGoalUsingHermite(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        search_frame->getCudaPtr(),
        search_frame->getCudaClassCostsPtr(),
        search_frame->getCudaFrameParamsPtr(),
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
