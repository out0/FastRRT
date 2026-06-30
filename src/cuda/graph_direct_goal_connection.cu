

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

__device__ __host__ float4 checkDirectConnectionToGoal(float4 *graphData, float3 *frame,
                                                       float *classCosts, int *searchSpaceParams, float max_curvature,
                                                       int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading,
                                                       bool isSafeZoneChecked, bool isDistanceToGoalProcessed,
                                                       float distance_to_goal_tolerance,
                                                       float max_heading_error)
{
    const int width = searchSpaceParams[FRAME_PARAM_WIDTH];
    const int height = searchSpaceParams[FRAME_PARAM_HEIGHT];
    const int minDistX = searchSpaceParams[FRAME_PARAM_MIN_DIST_X];
    const int minDistZ = searchSpaceParams[FRAME_PARAM_MIN_DIST_Z];

    // if (x == 128 && z == 128)
    //     printf ("checkDirectConnectionToGoal: minDistX, minDistZ = %d, %d\n", minDistX, minDistZ);

    const long pos = computePos(width, x, z);
    const float max_dist_to_goal_squared = distance_to_goal_tolerance * distance_to_goal_tolerance;

    float distance = frame[pos].y;
    if (!isDistanceToGoalProcessed)
    {
        const int dx = goal_x - x;
        const int dz = goal_z - z;
        distance = sqrtf(dx * dx + dz * dz);
    }

    int numPoints = TO_INT(1.5 * distance);

    float a1 = local_heading - HALF_PI;
    float a2 = goal_heading - HALF_PI;

    // Tangent vectors
    float2 tan1 = {distance * cosf(a1), distance * sinf(a1)};
    float2 tan2 = {distance * cosf(a2), distance * sinf(a2)};

    int last_x = -1;
    int last_z = -1;
    float last_heading = 0;

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

        last_heading = atan2f(ddz, ddx) + HALF_PI;

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
                return {-1, -1, 0, 0.0};
                //return;
            }
        }

        // Interpolated point
        last_x = cx;
        last_z = cz;

        // if (x == 124 && z == 112) {
        //     printf ("last_x = %d, last_z = %d\n", last_x, last_z);
        // }

        if (!__computeFeasibleForAngle(frame, searchSpaceParams, classCosts, minDistX, minDistZ, last_x, last_z, last_heading))
        {
            //  #ifndef __CUDA_ARCH__
            //  printf("[direct goal] %d,%d,%f --> %d,%d,%f not feasible\n",
            //              x, z, local_heading, goal_x, goal_z, goal_heading);
            // #endif
            // if (x == 128 && z == 128)
            //     printf("[CUDA] %d,%d,%f --> %d,%d,%f collision\n", x, z, local_heading, goal_x, goal_z, goal_heading);
            return {-1, -1, 0, 0.0};
            
        }
    }

    if (numPoints <= 0)
    {
        // if (x == 128 && z == 128)
        //      printf("[CUDA] %d,%d,%f --> %d,%d,%f numPoints <= 0\n", x, z, local_heading, goal_x, goal_z, goal_heading);
        return {-1, -1, 0, 0.0};
    }

    if (abs(last_heading - goal_heading) > max_heading_error)
        return {-1, -1, 0, 0.0};

    float dx = goal_x - last_x;
    float dz = goal_z - last_z;
    if ((dx * dx + dz * dz) > max_dist_to_goal_squared)
        return {-1, -1, 0, 0.0};

    
//    printf ("found a direct connection to goal on %d, %d, %f  min dist: %d, %d\n", last_x, last_z, last_heading, minDistX, minDistZ);
    return {TO_FLOAT(last_x), TO_FLOAT(last_z), last_heading, nodeCost};
}
