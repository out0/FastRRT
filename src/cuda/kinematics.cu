
#include <math_constants.h>
#include <string>
#include <cuda_runtime.h>
#include <driveless/cuda_params.h>
#include <driveless/math_utils.h>
#include "../../include/graph.h"

extern __device__ __host__ bool set(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);
extern __device__ __host__ int2 getParentCuda(int4 *graph, long pos);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ float getFrameCostCuda(float3 *frame, float *classCost, long pos);
extern __device__ __host__ float getIntrinsicCost(float4 *graphData, int width, int x, int z);

/// @brief Converts any map coordinate (x, y) to waypoint (x, z) assuming that location = (x = 0, y = 0, heading = 0)
/// @param center
/// @param rate_w
/// @param rate_h
/// @param coord
/// @return waypoint(x, z)
__device__ __host__ int2 convert_map_pose_to_waypoint(float3 *coord_origin, double2 coord)
{
    // map to waypoint formula is:
    //
    // [x' y' 1] = [x y 1] @ TranslationMatrix(lx, ly) @ RotationMatrix(-heading) @ ResizeMatrix(rh, rw)
    // x = Xcenter + y'
    // z = Zcenter - x'
    //
    // Assuming lx, ly = 0, 0  and heading = 0,  the TranslationMatrix and RotationMatrix become identity matrix.
    //
    // [x' y' 1] = [x y 1] @ ResizeMatrix(rh, rw)
    //
    // x' = x * rh
    // y' = y * rw
    //
    // x = Xcenter + y * rw
    // z = Zcenter - x * rh

    return {
        TO_INT((*coord_origin).x + coord.y),
        TO_INT((*coord_origin).y - coord.x)};
}

/// @brief Converts any waypoint (x, z) to map coordinate (x, y) assuming that location = (x = 0, y = 0, heading = 0)
/// @param center
/// @param rate_w
/// @param rate_h
/// @param coord
/// @return waypoint(x, z)
__device__ __host__ inline double2 convert_waypoint_to_map_pose(float3 *coord_origin, int2 coord)
{
    return {
        ((*coord_origin).y - coord.y),
        (coord.x - (*coord_origin).x)};
}

__device__ __host__ double compute_euclidean_2d_dist(const double2 &start, const double2 &end)
{
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ __host__ double compute_euclidean_2d_dist(const int2 &start, const int2 &end)
{
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ __host__ double compute_path_heading(const double2 p1, const double2 p2)
{
    double dy = p2.y - p1.y;
    double dx = p2.x - p1.x;

    if (dy >= 0 && dx > 0) // Q1
        return atan(dy / dx);
    else if (dy >= 0 && dx < 0) // Q2
        return CUDART_PI - atan(dy / abs(dx));
    else if (dy < 0 && dx > 0) // Q3
        return -atan(abs(dy) / dx);
    else if (dy < 0 && dx < 0) // Q4
        return atan(dy / dx) - CUDART_PI;
    else if (dx == 0 && dy > 0)
        return CUDART_PIO2_HI;
    else if (dx == 0 && dy < 0)
        return -CUDART_PIO2_HI;
    return 0.0;
}
__device__ __host__ double compute_path_heading(pose p1, pose p2)
{
    double dy = p2.y - p1.y;
    double dx = p2.x - p1.x;

    if (dy >= 0 && dx > 0) // Q1
        return atan(dy / dx);
    else if (dy >= 0 && dx < 0) // Q2
        return CUDART_PI - atan(dy / abs(dx));
    else if (dy < 0 && dx > 0) // Q3
        return -atan(abs(dy) / dx);
    else if (dy < 0 && dx < 0) // Q4
        return atan(dy / dx) - CUDART_PI;
    else if (dx == 0 && dy > 0)
        return CUDART_PIO2_HI;
    else if (dx == 0 && dy < 0)
        return -CUDART_PIO2_HI;
    return 0.0;
}
__device__ __host__ inline double clip(double val, double min, double max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

#define MIN_PATH_SIZE 1
__device__ __host__ float4 check_kinematic_new_path(int4 *graph, float4 *graphData,
                                                    double *physicalParams, int *searchSpaceParams, float3 *frame, float *classCosts,
                                                    float3 *ogStart, int2 start, float steeringAngle, float pathSize, float velocity_m_s)
{
    if (physicalParams == nullptr)
    {
        printf("[Error] the physical parameters are not set\n");
        return {-1, -1};
    }

    // const double rateW = physicalParams[PHYSICAL_PARAMS_RATE_W];
    // const double rateH = physicalParams[PHYSICAL_PARAMS_RATE_H];
    // const double invRateW = physicalParams[PHYSICAL_PARAMS_INV_RATE_W];
    // const double invRateH = physicalParams[PHYSICAL_PARAMS_INV_RATE_H];
    const double maxSteering = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];
    const double lr = physicalParams[PHYSICAL_PARAMS_LR];
    const int minDistX = searchSpaceParams[FRAME_PARAM_MIN_DIST_X];
    const int minDistZ = searchSpaceParams[FRAME_PARAM_MIN_DIST_Z];
    const int width = searchSpaceParams[FRAME_PARAM_WIDTH];
    const int height = searchSpaceParams[FRAME_PARAM_HEIGHT];

    const double2 startPose = convert_waypoint_to_map_pose(ogStart, start);
    // printf ("[check_kinematic_new_path] ogStart: %f, %f, %f\n", ogStart->x, ogStart->y, ogStart->z);
    // printf ("[check_kinematic_new_path] startPose: waypoint (%d, %d) -> map (%f, %f)\n", start.x, start.y, startPose.x, startPose.y);

    float x = startPose.x;
    float y = startPose.y;

    // printf("rateW: %f, rateH: %f, invRateW: %f, invRateH: %f, maxSteering: %f, lr: %f, startPose: (%d, %d)==(%f, %f)\n",
    //         rateW, rateH, invRateW, invRateH, maxSteering, lr, start.x, start.y, startPose.x, startPose.y);

    if (steeringAngle > maxSteering)
        steeringAngle = maxSteering;
    else if (steeringAngle < -maxSteering)
        steeringAngle = -maxSteering;

    const float steer = tanf(steeringAngle);
    const float dt = 0.1;
    const float ds = velocity_m_s * dt;
    const float beta = atanf(steer / 2);
    const float heading_increment_factor = ds * cosf(beta) * steer / (2 * lr);

    int maxSize = TO_INT(pathSize) + 1;

    int size = 0;

    int last_x = start.x;
    int last_z = start.y;
    long startPos = computePos(width, start.x, start.y);

    float heading = getHeadingCuda(graphData, startPos);
    int2 lastp;

    const float parentCost = getCostCuda(graphData, startPos);
    float nodeCost = parentCost;

    // int2 debug[5000];
    // int k = 0;

    int loop_count = 0;
    const int loop_max = (100 * maxSize) / dt;
    while (size < maxSize)
    {
        // printf ("current size: %d,  maxSize: %d, loop count: %d > %d?\n", size, maxSize,  loop_count, loop_max);
        if (loop_count++ > loop_max)
            break;
        x += ds * cosf(heading + beta);
        y += ds * sinf(heading + beta);
        heading += heading_increment_factor;

        lastp = convert_map_pose_to_waypoint(ogStart, {x, y});
        // printf ("next pose: map (%f, %f) -> (%d, %d) waypoint\n", x, y, lastp.x, lastp.y);

        if (lastp.x == last_x && lastp.y == last_z)
            continue;

        if (lastp.x < 0 || lastp.x >= width)
            break;

        if (lastp.y < 0 || lastp.y >= height)
            break;

        size += 1;
        nodeCost += getIntrinsicCost(graphData, width, lastp.x, lastp.y) + 1;

        if (!__computeFeasibleForAngle(frame, searchSpaceParams, classCosts, minDistX, minDistZ, lastp.x, lastp.y, heading))
        {
            // printf("unfeasible path from %d, %d to %d, %d\n", start.x, start.y, lastp.x, lastp.y);
            // printf ("path size: %d velocity_m_s: %f\n", size, velocity_m_s);
            return {-1.0, -1.0, 0.0, 0.0};
        }

        last_x = lastp.x;
        last_z = lastp.y;

        // debug[k++] = { last_x, last_z};

        // printf("x = %f, y = %f\n", x, y);
    }

    // printf("got out of the loop\n");

    if (size < MIN_PATH_SIZE)
        return {-1, -1};

    // printf ("path size: %d\n", size);

    return {(float)last_x, (float)last_z, nodeCost, heading};
}

__device__ __host__ bool check_kinematic_connection_start_end(int4 *graph, float4 *graphData, float3 *frame, double *physicalParams, int *searchSpaceParams, float *classCost, float3 *ogStart, int2 start, int2 end, float velocity_m_s, double &final_heading, double &path_cost)
{
    double distance = compute_euclidean_2d_dist(start, end);

    // const double rateW = physicalParams[PHYSICAL_PARAMS_RATE_W];
    // const double rateH = physicalParams[PHYSICAL_PARAMS_RATE_H];
    // const double invRateW = physicalParams[PHYSICAL_PARAMS_INV_RATE_W];
    // const double invRateH = physicalParams[PHYSICAL_PARAMS_INV_RATE_H];
    const double lr = physicalParams[PHYSICAL_PARAMS_LR];
    const double maxSteering = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];
    const int minDistX = searchSpaceParams[FRAME_PARAM_MIN_DIST_X];
    const int minDistZ = searchSpaceParams[FRAME_PARAM_MIN_DIST_Z];
    const int width = searchSpaceParams[FRAME_PARAM_WIDTH];

    double2 startM = convert_waypoint_to_map_pose(ogStart, start);
    double2 endM = convert_waypoint_to_map_pose(ogStart, end);
    double dt = 0.1;

    long startPos = computePos(width, start.x, start.y);
    double heading = getHeadingCuda(graphData, startPos);

    double path_heading = compute_path_heading(startM, endM);
    double steering_angle_deg = clip(path_heading - heading, -maxSteering, maxSteering);
    double ds = velocity_m_s * dt;

    int total_steps = TO_INT(distance / ds);

    double2 nextpM;
    nextpM.x = startM.x;
    nextpM.y = startM.y;
    int2 nextp, lastp = {start.x, start.y};

    double bestEndDist = distance;
    final_heading = 0;
    path_cost = 0;

    float curr_cost = 0;
    for (int i = 0; i < total_steps; i++)
    {
        double steer = tan(steering_angle_deg);
        double beta = atan(steer / lr);

        nextpM.x += ds * cos(heading + beta);
        nextpM.y += ds * sin(heading + beta);
        heading += ds * cos(beta) * steer / (2 * lr);

        path_heading = compute_path_heading(nextpM, endM);
        steering_angle_deg = clip(path_heading - heading, -maxSteering, maxSteering);
        double dist = compute_euclidean_2d_dist(nextpM, endM);

        nextp = convert_map_pose_to_waypoint(ogStart, nextpM);

        if (nextp.x == lastp.x && nextp.y == lastp.y)
            continue;

        curr_cost += getIntrinsicCost(graphData, width, nextp.x, nextp.y) + 1;

        if (bestEndDist < dist)
        {
            final_heading = heading;
            path_cost = curr_cost;
            return bestEndDist <= 2;
        }

        if (!__computeFeasibleForAngle(frame, searchSpaceParams, classCost, minDistX, minDistZ, lastp.x, lastp.y, heading))
            return false;

        lastp.x = nextp.x;
        lastp.y = nextp.y;
        bestEndDist = dist;
    }

    return true;
}

bool CudaGraph::checkFeasibleConnection(float3 *og, int2 init, int2 end, int velocity_m_s)
{
    double finalHeading = 0;
    double pathCost = 0;

    return check_kinematic_connection_start_end(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _physicalParams->get(),
        _searchSpaceParams->get(),
        _classCosts->get(),
        _ogCoordinateStart->get(),
        init,
        end,
        velocity_m_s,
        finalHeading,
        pathCost);
}

__device__ __host__ bool check_graph_connection_with_hermite(
    int4 *graph,
    float4 *graphData,
    float3 *frame,
    double *physicalParams,
    int *searchSpaceParams,
    float *classCost,
    int2 start,
    int2 end,
    float velocity_m_s,
    float &path_cost)
{

    const int numPoints = abs(end.y - start.y);
    if (numPoints < 2)
    {
        return false;
    }
    const int width = searchSpaceParams[FRAME_PARAM_WIDTH];
    const int height = searchSpaceParams[FRAME_PARAM_HEIGHT];
    const int minDistX = searchSpaceParams[FRAME_PARAM_MIN_DIST_X];
    const int minDistZ = searchSpaceParams[FRAME_PARAM_MIN_DIST_Z];

    // Distance between points (used to scale tangents)
    const float dx = end.x - start.x;
    const float dz = end.y - start.y;
    const float d = sqrtf(dx * dx + dz * dz);
    const float a1 = getHeadingCuda(graphData, computePos(width, start.x, start.y)) - HALF_PI;
    const float a2 = getHeadingCuda(graphData, computePos(width, end.x, end.y)) - HALF_PI;
    // Tangent vectors
    const float2 tan1 = {d * cosf(a1), d * sinf(a1)};
    const float2 tan2 = {d * cosf(a2), d * sinf(a2)};

    int last_x = -1;
    int last_z = -1;
    path_cost = 0;

    for (int i = 0; i < numPoints; ++i)
    {
        float t = (float)i / (numPoints - 1);

        float t2 = t * t;
        float t3 = t2 * t;

        // Hermite basis functions
        float h00 = 2 * t3 - 3 * t2 + 1;
        float h10 = t3 - 2 * t2 + t;
        float h01 = -2 * t3 + 3 * t2;
        float h11 = t3 - t2;

        float x = h00 * start.x + h10 * tan1.x + h01 * end.x + h11 * tan2.x;
        float z = h00 * start.y + h10 * tan1.y + h01 * end.y + h11 * tan2.y;

        if (x < 0 || x >= width)
            continue;
        if (z < 0 || z >= height)
            continue;

        int cx = TO_INT(round(x));
        int cz = TO_INT(round(z));

        if (cx == last_x && cz == last_z)
            continue;

        float t00 = 6 * t2 - 6 * t;
        float t10 = 3 * t2 - 4 * t + 1;
        float t01 = -6 * t2 + 6 * t;
        float t11 = 3 * t2 - 2 * t;

        float ddx = t00 * start.x + t10 * tan1.x + t01 * end.x + t11 * tan2.x;
        float ddz = t00 * start.y + t10 * tan1.y + t01 * end.y + t11 * tan2.y;

        float heading = atan2f(ddz, ddx) + HALF_PI;

        if (!__computeFeasibleForAngle(frame, searchSpaceParams, classCost, minDistX, minDistZ, cx, cz, heading))
            return false;

        last_x = cx;
        last_z = cz;
        path_cost += getIntrinsicCost(graphData, width, cx, cz) + 1;
    }

    return true;
}