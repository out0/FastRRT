

#include "../../include/graph.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ float getIntrinsicCost(float4 *graphData, int width, int x, int z);
extern __device__ __host__ double computeHeading(int x1, int z1, int x2, int z2);
__device__ __host__ bool check_bit(int traversability, int bit)
{
    return (traversability & bit) > 0;
}

__device__ __host__ float checkDirectConnectionToGoal(float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading, bool isSafeZoneChecked, bool isDistanceToGoalProcessed)
{
    const int width = searchSpaceParams[FRAME_PARAM_WIDTH];
    const int height = searchSpaceParams[FRAME_PARAM_HEIGHT];
    const int minDistX = searchSpaceParams[FRAME_PARAM_MIN_DIST_X];
    const int minDistZ = searchSpaceParams[FRAME_PARAM_MIN_DIST_Z];

    // if (x == 128 && z == 128)
    //     printf ("checkDirectConnectionToGoal: minDistX, minDistZ = %d, %d\n", minDistX, minDistZ);

    const long pos = computePos(width, x, z);

    float distance = 0.0;
    if (isDistanceToGoalProcessed)
    {
        distance = frame[pos].y;
    }
    else
    {
        const int dx = goal_x - x;
        const int dz = goal_z - z;
        distance = sqrtf(dx * dx + dz * dz);
    }

    int numPoints = TO_INT(distance);

    float a1 = local_heading - HALF_PI;
    float a2 = goal_heading - HALF_PI;

    // Tangent vectors
    float2 tan1 = {distance * cosf(a1), distance * sinf(a1)};
    float2 tan2 = {distance * cosf(a2), distance * sinf(a2)};

    int last_x = -1;
    int last_z = -1;

    const float parentCost = getCostCuda(graphData, pos);
    float nodeCost = parentCost;

    for (int i = 0; i < numPoints; ++i)
    {
        double t = ((double)0.0 + i) / (numPoints - 1);

        double t2 = t * t;
        double t3 = t2 * t;

        // Hermite basis functions
        double h00 = 2 * t3 - 3 * t2 + 1;
        double h10 = t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 = t3 - t2;

        double px = h00 * x + h10 * tan1.x + h01 * goal_x + h11 * tan2.x;
        double pz = h00 * z + h10 * tan1.y + h01 * goal_z + h11 * tan2.y;

        if (px < 0 || px >= width)
            continue;
        if (pz < 0 || pz >= height)
            continue;

        int cx = TO_INT(px);
        int cz = TO_INT(pz);

        if (cx == last_x && cz == last_z)
            continue;
        if (cx < 0 || cx >= width)
            continue;
        if (cz < 0 || cz >= height)
            continue;

        nodeCost += getIntrinsicCost(graphData, width, cx, cz) + 1;

        double t00 = 6 * t2 - 6 * t;
        double t10 = 3 * t2 - 4 * t + 1;
        double t01 = -6 * t2 + 6 * t;
        double t11 = 3 * t2 - 2 * t;

        double ddx = t00 * x + t10 * tan1.x + t01 * goal_x + t11 * tan2.x;
        double ddz = t00 * z + t10 * tan1.y + t01 * goal_z + t11 * tan2.y;

        float heading = atan2f(ddz, ddx) + HALF_PI;

        double d00 = 12 * t - 6;
        double d10 = 6 * t - 4;
        double d01 = -12 * t + 6;
        double d11 = 6 * t - 2;

        double dd2x = d00 * x + d10 * tan1.x + d01 * goal_x + d11 * tan2.x;
        double dd2z = d00 * z + d10 * tan1.y + d01 * goal_z + d11 * tan2.y;

        if (max_curvature > 0)
        {
            float k = abs(ddx * dd2z - ddz * dd2x) / pow(ddx * ddx + ddz * ddz, 1.5);
            if (k > max_curvature)
            {
                // if (x == 128 && z == 128)
                // #ifndef __CUDA_ARCH__
                //      printf("[direct goal] %d,%d,%f --> %d,%d,%f max curvature excedded: %f (max %f)\n",
                //          x, z, local_heading, goal_x, goal_z, goal_heading, k, max_curvature);
                // #endif
                return -1;
            }
        }

        // Interpolated point
        last_x = cx;
        last_z = cz;

        int traversability = TO_INT(frame[pos].z);

        // if (x == 124 && z == 112) {
        //     printf ("last_x = %d, last_z = %d\n", last_x, last_z);
        // }

        if (isSafeZoneChecked && check_bit(traversability, 0x100)) {
            // if (x == 128 && z == 128)
            //      printf ("SAFEZONE CHECKED last_x = %d, last_z = %d\n", last_x, last_z);
            // }
            continue;
        }

        if (!__computeFeasibleForAngle(frame, searchSpaceParams, classCosts, minDistX, minDistZ, last_x, last_z, heading))
        {
            //  #ifndef __CUDA_ARCH__
            //  printf("[direct goal] %d,%d,%f --> %d,%d,%f not feasible\n",
            //              x, z, local_heading, goal_x, goal_z, goal_heading);
            // #endif
            // if (x == 128 && z == 128)
            //     printf("[CUDA] %d,%d,%f --> %d,%d,%f collision\n", x, z, local_heading, goal_x, goal_z, goal_heading);
            return -1;
        }
    }

    if (numPoints <= 0) {
        // if (x == 128 && z == 128)
        //      printf("[CUDA] %d,%d,%f --> %d,%d,%f numPoints <= 0\n", x, z, local_heading, goal_x, goal_z, goal_heading);
        return -1;
    }

    if (last_x != goal_x && last_z != goal_z) {
        // #ifndef __CUDA_ARCH__
        //  printf("[direct goal] %d,%d,%f --> %d,%d,%f goal not reached\n",
        //                  x, z, local_heading, goal_x, goal_z, goal_heading);

        // #endif
        return -1;
    }

    // if (x == 128 && z == 128)
    //     printf("[CUDA] %d,%d connects to %d,%d,, goal %d, %d\n", x, z , last_x, last_z, goal_x, goal_z);
    return nodeCost;
}

__global__ static void __CUDA_direct_connection(int4 *graph, float4 *graphData, float3 *frame, float3 *directConnectData, float *classCosts, int *searchSpaceParams,
                                                float max_curvature, int goal_x, int goal_z, float goal_heading, bool isSafeZoneChecked, bool isDistanceToGoalProcessed)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    const int width = searchSpaceParams[FRAME_PARAM_WIDTH];
    const int height = searchSpaceParams[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    directConnectData[pos].x = 0;

    int z = pos / width;
    int x = pos - z * width;

    float local_heading = computeHeading(x, z, goal_x, goal_z);

    float cost = checkDirectConnectionToGoal(graphData, frame, classCosts, searchSpaceParams, max_curvature, x, z, local_heading, goal_x, goal_z, goal_heading, isSafeZoneChecked, isDistanceToGoalProcessed);

    if (cost < 0)
        return;

    directConnectData[pos].x = 1.0;
    directConnectData[pos].y = local_heading;
    directConnectData[pos].z = cost;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    // temp
    //printf ("%d, %d connects directly to goal %d, %d with heading %f deg\n", x, z, goal_x, goal_z, 180 * local_heading / PI);
}

void CudaGraph::processDirectGoalConnection(SearchFrame *frame, int goal_x, int goal_z, angle goal_heading, float max_curvature)
{
    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    if (max_curvature < 0)
        max_curvature = _physicalParams->get()[PHYSICAL_MAX_CURVATURE];

    //printf ("minx, minz = %d, %d\n",_searchSpaceParams->get()[FRAME_PARAM_MIN_DIST_X], _searchSpaceParams->get()[FRAME_PARAM_MIN_DIST_Z]);

    __CUDA_direct_connection<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        frame->getCudaPtr(),
        _graphGoalDirectConnection->getCudaPtr(),
        frame->getCudaClassCostsPtr(),
        frame->getCudaFrameParamsPtr(),
        max_curvature,
        goal_x,
        goal_z,
        goal_heading.rad(),
        false, false);
        //frame->isSafeZoneChecked(),
        //frame->isDistanceToGoalProcessed());

    CUDA(cudaDeviceSynchronize());
}

__device__ __host__ bool is_directly_connected_to_goal(float3 *goalDirectConnectionData, int width, int x, int z) {
    long pos = computePos(width, x, z);
    return goalDirectConnectionData[pos].x > 0;
}

bool CudaGraph::isDirectlyConnectedToGoal(int x, int z)
{
    return is_directly_connected_to_goal(_graphGoalDirectConnection->getCudaPtr(), _graph->width(), x, z);
}

__device__ __host__ float get_cost_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z) {
    long pos = computePos(width, x, z);
    return goalDirectConnectionData[pos].z;
}

float CudaGraph::directConnectionToGoalCost(int x, int z)
{
    if (!is_directly_connected_to_goal(_graphGoalDirectConnection->getCudaPtr(), _graph->width(), x, z))
        return -1;
    
    return get_cost_direct_connection_to_goal(_graphGoalDirectConnection->getCudaPtr(), _graph->width(), x, z);
}

__device__ __host__ float get_heading_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z) {
    long pos = computePos(width, x, z);
    return goalDirectConnectionData[pos].y;
}

angle CudaGraph::directConnectionToGoalHeading(int x, int z)
{
    float h = get_heading_direct_connection_to_goal(_graphGoalDirectConnection->getCudaPtr(), _graph->width(), x, z);
    return angle::rad(h);
}