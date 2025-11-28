#include <driveless/cuda_params.h>
#include <driveless/math_utils.h>
#include <stdio.h>

extern __device__ __host__ long computePos(int width, int x, int z);

__device__ __host__ float getFrameCostCuda(float3 *frame, float *classCost, long pos)
{
    int segmentation_class = TO_INT(frame[pos].x);
    return classCost[segmentation_class];
}

__device__ __host__ bool isAllAngleFeasible(float3 *frame, long pos)
{
    return TO_INT(frame[pos].z) & 256 > 0;
}

__device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    //printf("computeFeasibleForAngle [1]\n");

    long pos = computePos(width, x, z);

    if (isAllAngleFeasible(frame, pos))
    {
        // if (x == 127 && z == 100)
        //     printf("computeFeasibleForAngle: feasible for all at %d, %d\n", x, z);

        return true;
    }

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    //printf("minDistX: %d, minDistZ: %d\n", minDistX, minDistZ);

    //printf("computeFeasibleForAngle [2]\n");

    for (int i = -minDistZ; i <= minDistZ; i++)
        for (int j = -minDistX; j <= minDistX; j++)
        {
            int xl = TO_INT(j * c - i * s + x);
            int zl = TO_INT(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            int segmentation_class = TO_INT(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                // if (x == 127 && z == 100)
                //     printf("(%d, %d) invalid on %d, %d segmentation_class: %d (x param = %f) class cost %f\n", x, z, xl, zl, segmentation_class, frame[zl * width + xl].x, classCost[segmentation_class]);
                return false;
            }

            // if (x == 127 && z == 100) {
            //     printf ("%d, %d feasible while checking %d, %d minDistX: %d, minDistZ: %d\n", x, z, xl, zl, minDistX, minDistZ);
            // }
        }

    //printf("computeFeasibleForAngle %d, %d\n", x, z);
    return true;
}

__device__ __host__ bool checkStateFeasible(float3 *searchFrame, int *params, float *classCosts, float4 *waypoints, int waypoints_size, int current_pos_waypoints, int minDistX, int minDistZ)
{
    float heading;

    const int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    const int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    const int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    const int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];
    // const int width = params[FRAME_PARAM_WIDTH];
    // const int height = params[FRAME_PARAM_HEIGHT];

    int x = waypoints[current_pos_waypoints].x;
    int z = waypoints[current_pos_waypoints].y;

    waypoints[current_pos_waypoints].w = 1.0;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        return true;
    }

    // if (computeHeading)
    // {
    //     bool valid = true;
    //     heading = ___computeMeanHeading(waypoints, current_pos_waypoints, waypoints_size, &valid, width, height);
    //     waypoints[current_pos_waypoints].z = heading;
    // }
    // else
    heading = waypoints[current_pos_waypoints].z;
    // printf ("heading: %f\n", heading);

    bool res = __computeFeasibleForAngle(searchFrame, params, classCosts, minDistX, minDistZ, x, z, heading);

    if (!res)
        waypoints[current_pos_waypoints].w = 0.0;

    return res;
}
