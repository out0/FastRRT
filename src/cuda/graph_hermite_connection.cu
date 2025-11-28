
#include "../../include/graph.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ float getIntrinsicCost(float4 *graphData, int width, int x, int z);
extern __device__ __host__ bool set(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);



__device__ __host__ float cudaHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, int goal_x, int goal_z, float goal_heading, bool fillInterpolation, bool computeDistance)
{
    // int numPoints = 2 * abs(max(int(p2.z() - p1.z()), int(p2.x() - p1.x()), 100));
    const int width = searchSpaceParams[FRAME_PARAM_WIDTH];
    const int height = searchSpaceParams[FRAME_PARAM_HEIGHT];
    const int minDistX = searchSpaceParams[FRAME_PARAM_MIN_DIST_X];
    const int minDistZ = searchSpaceParams[FRAME_PARAM_MIN_DIST_Z];

    const long pos = computePos(width, x, z);
    float distance = frame[pos].y;
    
    if (computeDistance) {
        const float dx = goal_x - x;
        const float dz = goal_z - z;
        distance = sqrtf(dx * dx + dz * dz);
    }
    
    int numPoints = TO_INT(distance);

    float local_heading = getHeadingCuda(graphData, pos);

    float a1 = local_heading - PI / 2;
    float a2 = goal_heading - PI / 2;

    // Tangent vectors
    float2 tan1 = {distance * cosf(a1), distance * sinf(a1)};
    float2 tan2 = {distance * cosf(a2), distance * sinf(a2)};

    int last_x = x;
    int last_z = z;

    const float parentCost = getCostCuda(graphData, pos);
    float nodeCost = parentCost;

    // float dx = goal_x - x;
    // float dz = goal_z - z;

    // printf ("dist: %f, computed dist: %f, numPoints = %d\n", distance, sqrtf(dx*dx + dz*dz),  numPoints);

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
            float k = abs(ddx * dd2z - ddz * dd2x) / pow(ddx * ddx + ddz * ddz, 3 / 2);
            if (k > max_curvature) {
                // printf ("[CUDA] %d,%d,%f --> %d,%d,%f max curvature excedded: %f (max %f)\n",
                // x, z, local_heading, goal_x, goal_z, goal_heading, k, max_curvature);
                return -2;
            }
        }

        // Interpolated point
        if (fillInterpolation) {
            
            // if (last_x == -1) {
            //     set(graph, graphData, computePos(width, cx, cz), heading, x, z, nodeCost, GRAPH_TYPE_NODE, true);
            //     printf ("adding %d, %d, %f with parent = %d, %d\n", cx, cz, heading, x, z);
            // }
            // else {
                set(graph, graphData, computePos(width, cx, cz), heading, last_x, last_z, nodeCost, GRAPH_TYPE_NODE, true);
                printf ("adding %d, %d, %f with parent = %d, %d\n", cx, cz, heading, last_x, last_z);
            // }
            last_x = cx;
            last_z = cz;
            continue;
        }

        last_x = cx;
        last_z = cz;

        if (!__computeFeasibleForAngle(frame, searchSpaceParams, classCosts, minDistX, minDistZ, last_x, last_z, heading)) {
            // printf ("[CUDA] %d,%d,%f --> %d,%d,%f collision\n",
            //     x, z, local_heading, goal_x, goal_z, goal_heading);
            return -1;
        }
    }

    if (numPoints <= 0) return -1;
    return nodeCost;
}

__device__ __host__ float canConnectToGoalUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, int goal_x, int goal_z, float goal_heading) {
    return cudaHermite(graph, graphData, frame, classCosts, searchSpaceParams, max_curvature, x, z, goal_x, goal_z, goal_heading, false, false);
}

__device__ __host__ float interpolateNodesUsingHermite(int4 *graph, float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, int goal_x, int goal_z, float goal_heading) {
    return cudaHermite(graph, graphData, frame, classCosts, searchSpaceParams, max_curvature, x, z, goal_x, goal_z, goal_heading, true, true);
}