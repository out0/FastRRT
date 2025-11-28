
#include "../../include/graph.h"
#include "../../include/cuda_params.h"

extern __device__ __host__ bool check_graph_connection_with_hermite(
    int4 *graph, 
    float3 *graphData, 
    float3 *frame, 
    double *physicalParams, 
    int *params, 
    float *classCost, 
    int2 center, 
    int2 start, 
    int2 end, 
    float velocity_m_s,
    float &path_cost);

double computeHeading(int x1, int z1, int x2, int z2)
{
    double dz = z2 - z1;
    double dx = x2 - x1;

    if (dx == 0 && dz == 0)
        return 0;

    double v1 = 0;
    if (dz != 0)
        v1 = atan2(-dz, dx);
    else
        v1 = atan2(0, dx);

    return HALF_PI - v1;
}

void CudaGraph::optimizeGraph(float3 *og, angle goalHeading, float radius, float velocity_m_s)
{
    // Generate node candidates
    expandTree(og, goalHeading, radius, velocity_m_s, false);

    int numNodesInGraph = count();

    // list all feasible node candidates
    std::pair<int2 *, int> lst = __listNodes(GRAPH_TYPE_TEMP);

    // printf("Derivating %d nodes\n", lst.second);

    // for (int i = 0; i < lst.second; i++)
    // {
    //     int2 node = lst.first[i];
    //     int2 parent = getParent(node.x, node.y);
    //     if (parent.x != -1 && parent.y != -1)
    //         printf("Node %d: (%d, %d), parent (%d, %d), cost: %f, heading: %f\n", i, node.x, node.y, parent.x, parent.y,
    //                getCost(node.x, node.y), getHeading(node.x, node.y).deg());
    // }

    int count = lst.second;
    int2 *cudaResult = lst.first;

    numNodesInGraph += count; // add candidates

    for (int i = 0; i < count; i++)
    {
        optimizeNode(og, cudaResult[i].x, cudaResult[i].y, radius, velocity_m_s, numNodesInGraph);
    }

    cudaFreeHost(cudaResult);
    acceptDerivedNodes();   
}

extern __device__ __host__ int2 getParentCuda(int4 *graph, long pos);
extern __device__ __host__ void setParentCuda(int4 *graph, long pos, int parent_x, int parent_z);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getCostCuda(float3 *graphData, long pos);
extern __device__ __host__ float getHeadingCuda(float3 *graphData, long pos);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);

__device__ __host__ bool checkCyclicRef(int4 *graph, int width, int height, int x, int z, int xc, int zc, int numNodesInGraph)
{

    int xi = xc;
    int zi = zc;
    for (int i = 0; i <= numNodesInGraph; i++)
    {
        int2 parent = getParentCuda(graph, computePos(width, xi, zi));
        if (parent.x == x && parent.y == z)
            return true; // we've reached (x,z) from (xc, zc), so there's a cyclic ref.
        if (parent.x == -1 && parent.y == -1)
            return false; // we've reached the origin
        xi = parent.x;
        zi = parent.y;
    }
    // the halt problem is avoided by returning true if we hopped for more than numNodesInGraph:
    return true;
}

__device__ __host__ void __node_optimize(
    int pos,
    int4 *graph,
    float3 *graphData,
    float3 *frame,
    double *physicalParams,
    int *params,
    float *classCost,
    int2 center,
    int xc,
    int zc,
    float radius_sqr,
    float velocity_m_s,
    int numNodesInGraph)
{

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (graph[pos].z != GRAPH_TYPE_NODE)
        return;

    int z = pos / width;
    int x = pos - z * width;
    // exit if is the same node as the candidate
    if (x == xc && z == zc)
        return;

    // exit if node is parent
    int2 parent = getParentCuda(graph, computePos(width, xc, zc));
    if (parent.x == x && parent.y == z)
        return;

    // exit if not in range
    float dx = x - xc;
    float dz = z - zc;
    float dist_sq = dx * dx + dz * dz;
    if (dist_sq > radius_sqr)
        return;

    // check cyclic ref
    if (checkCyclicRef(graph, width, height, x, z, xc, zc, numNodesInGraph))
    {
        // printf("cyclic ref in (%d, %d) --> (%d, %d pos: %d)\n", x, z, xc, zc, pos);
        return;
    }

    float maxPathSize = 2 * sqrtf(dist_sq);

    // check if we can connect the node N in range with x,z as its parent, maintaining heading (x,z) as starting heading and heading_N as arriving reading
    //  and having total size <= maxPathSize

    float newPathCost = 0.0; // TODO

    if (!check_graph_connection_with_hermite(graph, graphData, frame,
                                physicalParams, params, classCost,
                                center,
                                {xc, zc},
                                {x, z},
                                velocity_m_s,
                                newPathCost))
    {
        // printf("this connection is not feasible (%d, %d) --> (%d, %d)\n", x, z, xc, zc);
        return;
    }

    float newCost = getCostCuda(graphData, computePos(width, xc, zc)) + newPathCost;
    float currentCost = getCostCuda(graphData, pos);

    if (newCost < currentCost)
    {
        // connect N to xc, zc
        setParentCuda(graph, pos, xc, zc);
//        setTypeCuda(graph, computePos(width, xc, zc), GRAPH_TYPE_NODE);
        // printf("new connection (%d, %d) --> (%d, %d)\n", x, z, xc, zc);
    }
    // else
    // {
    //     printf("new connection (%d, %d) --> (%d, %d) cost %f > %f\n", x, z, xc, zc, newCost, currentCost);
    // }
}

__global__ static void __CUDA_KERNEL_optimize(
    int4 *graph,
    float3 *graphData,
    float3 *frame,
    double *physicalParams,
    int *params,
    float *classCost,
    int2 center,
    int xc,
    int zc,
    float radius_sqr,
    float velocity_m_s,
    int numNodesInGraph)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    __node_optimize(pos, graph, graphData, frame, physicalParams,
                    params, classCost, center, xc, zc, radius_sqr, velocity_m_s, numNodesInGraph);
}

void CudaGraph::optimizeNode(float3 *og, int x, int z, float radius, float velocity_m_s, int numNodesInGraph)
{

    // for (int i = 0; i < 256; i++)
    //     for (int j = 0; j < 256; j++)
    //     {
    //         int pos = 256 * i + j;
    //         __node_optimize(pos,
    //                         _frame->getCudaPtr(),
    //                         _frameData->getCudaPtr(),
    //                         og,
    //                         _physicalParams,
    //                         _searchSpaceParams,
    //                         _classCosts,
    //                         _gridCenter,
    //                         x,
    //                         z,
    //                         radius * radius,
    //                         velocity_m_s,
    //                         numNodesInGraph);
    //     }

    // return;

    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_optimize<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frameData->getCudaPtr(),
        og,
        _physicalParams,
        _searchSpaceParams,
        _classCosts,
        _gridCenter,
        x,
        z,
        radius * radius,
        velocity_m_s,
        numNodesInGraph);

    CUDA(cudaDeviceSynchronize());
}
