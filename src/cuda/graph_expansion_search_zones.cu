
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../../include/graph.h"

extern __device__ __host__ float4 check_kinematic_new_path(int4 *graph, float4 *graphData, double *physicalParams, int *searchSpaceParams, float3 *frame, float *classCosts, float3 *ogStart, int2 start, float steeringAngle, float pathSize, float velocity_m_s);
extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
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
extern __device__ __host__ void decNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ int getNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ void setNodeDeriveCount(int4 *graph, long pos, int count);
extern __device__ __host__ float canConnectToGoalUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_steering_rad, int x, int z, int goal_x, int goal_z, float goal_heading);
extern __device__ __host__ void setDirectCostCuda(float4 *graphData, long pos, float cost);
extern __device__ __host__ void assertDAGconsistency(int4 *graph, float4 *graphData, int width, int height, long pos);
extern __device__ __host__ int2 expand_node(int4 *graph, float4 *graphData, float3 *frame, long pos, int x, int z, float steeringAngle_rad, float pathSize, float *classCosts, int *searchParams, double *physicalParams, float3 *ogStart, float velocity_m_s, bool *nodeCollision);
#define MIN_PATH_SIZE 5.0


#define BLOCK_SIZE 128
#define CHECK_NO_COLLISION 1

__device__ __host__ int computeDensityPos(int density_width, int x, int z)
{
    int density_x = TO_INT(x / BLOCK_SIZE);
    int density_z = TO_INT(z / BLOCK_SIZE);
    return (density_z * density_width + density_x);
}

__global__ void __CUDA_count_nodes_in_density_region(int4 *graph, int *params, unsigned int *node_count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int density_width = params[FRAME_DENSITY_WIDTH];

    if (pos >= width * height)
        return;

    int type = getTypeCuda(graph, pos);

    if (type == GRAPH_TYPE_NULL || type == GRAPH_TYPE_PROCESSING)
        return;

    int z = pos / width;
    int x = pos - z * width;

    const int densityPos = computeDensityPos(density_width, x, z);

    // printf ("%d, %d incrementing density region %d\n", x, z, densityPos);

    atomicInc(&node_count[densityPos], 99999999);
}

__device__ __host__ bool checkCanExpand(int4 *graph, unsigned int *region_count, int *params, float node_mean, int pos, int x, int z, bool expandFrontier)
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
    int width = _searchSpaceParams->get()[FRAME_PARAM_WIDTH];
    int height = _searchSpaceParams->get()[FRAME_PARAM_HEIGHT];

    int density_width = TO_INT(width / BLOCK_SIZE) + 1;
    int density_height = TO_INT(height / BLOCK_SIZE) + 1;
    int density_size = density_width * density_height;

    _searchSpaceParams->get()[FRAME_DENSITY_WIDTH] = density_width;
    _searchSpaceParams->get()[FRAME_DENSITY_HEIGHT] = density_height;
    _searchSpaceParams->get()[FRAME_DENSITY_SIZE] = density_size;

    // printf("graph size: %d, %d\n", width, height);
    // printf("num of density regions: %d\n", density_size);
    // printf("density region size: %d x %d\n", density_width, density_height);

    _region_node_count = std::make_unique<CudaPtr<unsigned int>>(density_size);
    for (int i = 0; i < density_size; i++)
    {
        _region_node_count->get()[i] = 0;
    }

    _node_mean = 0;
}

void CudaGraph::__dealocRegionDensity()
{
    _region_node_count = nullptr;
}

void CudaGraph::computeGraphRegionDensity()
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    int density_size = _searchSpaceParams->get()[FRAME_DENSITY_SIZE];

    for (int i = 0; i < density_size; i++)
    {
        _region_node_count->get()[i] = 0;
    }

    __CUDA_count_nodes_in_density_region<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _searchSpaceParams->get(),
        _region_node_count->get());

    CUDA(cudaDeviceSynchronize());

    _node_mean = 0;
    int numRegionsWithNodes = 0;
    for (int i = 0; i < density_size; i++)
    {
        _node_mean += _region_node_count->get()[i];
        if (_region_node_count->get()[i] > 0)
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
        if (_region_node_count->get()[i] == 0)
            continue;
        int density_x = TO_INT(i % _searchSpaceParams->get()[FRAME_DENSITY_WIDTH]);
        int density_z = TO_INT(i / _searchSpaceParams->get()[FRAME_DENSITY_WIDTH]);
        // printf("density region (%d, %d): %d\n", density_x, density_z, _region_node_count[i]);
    }
}

__global__ void __CUDA_smart_node_expansion(curandState *state, int4 *graph, float4 *graphData, float3 *frame, unsigned int *region_count, int node_mean, float *classCosts, int *searchParams, double *physicalParams, float3 *ogStart, float maxPathSize, float velocity_m_s, bool expandFrontier, bool forceExpand, bool *nodeCollision, int goal_x, int goal_z, float goal_heading)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = searchParams[FRAME_PARAM_WIDTH];
    int height = searchParams[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (!checkInGraphCuda(graph, pos))
        return;

    int z = pos / width;
    int x = pos - z * width;

    // Smart expansion: if this node is a leaf, it can be expanded. If not, it still can be expanded if the region density is lower than the mean density
    if (!forceExpand && !checkCanExpand(graph, region_count, searchParams, node_mean, pos, x, z, expandFrontier))
    {
        // const int densityPos = computeDensityPos(searchParams[FRAME_DENSITY_WIDTH], x, z);
        // printf("wont expand (%d, %d) because of density: %d vs mean %d\n", x, z, region_count[densityPos], node_mean);
        return;
    }

    float heading = getHeadingCuda(graphData, pos);
    double maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

    double steeringAngle = generateRandomNeg(state, pos, maxSteeringAngle);
    double pathSize = generateRandom(state, pos, 5.0, maxPathSize);
    if (pathSize <= 0)
        pathSize = MIN_PATH_SIZE;

    expand_node(graph, graphData, frame, pos, x, z, steeringAngle, pathSize, classCosts, searchParams, physicalParams, ogStart, velocity_m_s, nodeCollision);
}

void CudaGraph::smartExpansion(float3 *og, angle goalHeading, float maxPathSize, float velocity_m_s, bool expandFrontier, bool forceExpand, int2 goal, angle goal_heading)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_nodeCollision->get() = false;

    __CUDA_smart_node_expansion<<<numBlocks, THREADS_IN_BLOCK>>>(
        _randState->get(),
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _region_node_count->get(),
        _node_mean,
        _classCosts->get(),
        _searchSpaceParams->get(),
        _physicalParams->get(),
        _ogCoordinateStart->get(),
        maxPathSize,
        velocity_m_s,
        expandFrontier,
        forceExpand,
        _nodeCollision->get(),
        goal.x,
        goal.y,
        goal_heading.rad());

    CUDA(cudaDeviceSynchronize());

    computeGraphRegionDensity();

    if (*_nodeCollision->get())
    {
        //printf("solving graph collision\n");
        solveCollisions();
    }
}