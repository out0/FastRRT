#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include "../../include/cuda_graph.h"
#include <fstream>

extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ float4 check_kinematic_new_path(int4 *graph, float4 *graphData, double *physicalParams, int *searchSpaceParams, float3 *frame, float *classCosts, float3 *ogStart, int2 start, float steeringAngle, float pathSize, float velocity_m_s);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ void incNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ bool set(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);
extern __device__ __host__ void decNodeDeriveCount(int4 *graph, long pos);

__device__ __host__ bool change_graph_type_if_current_value_equals_expected_value(int4 *graph, long pos, int expected_value, int new_value)
{

#ifdef __CUDA_ARCH__
    return atomicCAS(&(graph[pos].z), expected_value, new_value) == expected_value;
#else
    if (graph[pos].z == expected_value)
    {
        graph[pos].z = new_value;
        return true;
    }
    return false;
#endif
}


__device__ __host__ float4 expand_node(int4 *graph, float4 *graphData, float3 *frame, long pos, int x, int z, float steeringAngle_rad,
                                       float pathSize, float *classCosts, int *searchParams, double *physicalParams, float3 *ogCoordinateStart, float velocity_m_s, bool *nodeCollision,
                                       bool ignore_collision)
{
    int width = searchParams[FRAME_PARAM_WIDTH];

    float heading = getHeadingCuda(graphData, pos);

    float4 end = check_kinematic_new_path(graph, graphData, physicalParams, searchParams, frame, classCosts, ogCoordinateStart, {x, z}, steeringAngle_rad, pathSize, velocity_m_s);

    // printf("end expansion: %f, %f, heading: %f, cost: %f\n", end.x, end.y, end.w, end.z);

    if (end.x < 0 || end.y < 0)
        return {-1, -1, -1, 0.0};

    int end_x = TO_INT(end.x);
    int end_z = TO_INT(end.y);

    if (end_x == ogCoordinateStart->x && end_z == ogCoordinateStart->y)
    {
        return {-1, -1, -1, 0.0};
    }

    float end_cost = end.z;
    float end_heading = end.w;

    long end_pos = computePos(width, end_x, end_z);

    if (end_pos == pos)
        return {-1, -1, -1, 0.0};

    if (change_graph_type_if_current_value_equals_expected_value(graph, end_pos, GRAPH_TYPE_NULL, GRAPH_TYPE_TEMP))
    {
        // A new node is being added to the graph
        incNodeDeriveCount(graph, pos);
        set(graph, graphData, end_pos, end_heading, x, z, end_cost, GRAPH_TYPE_TEMP, true);
        return {end.x, end.y, end_heading, 1.0};
    }

    if (!ignore_collision && change_graph_type_if_current_value_equals_expected_value(graph, end_pos, GRAPH_TYPE_NODE, GRAPH_TYPE_COLLISION))
    {
        set(graph, graphData, end_pos, end_heading, x, z, end_cost, GRAPH_TYPE_COLLISION, true);
        *nodeCollision = true;
        decNodeDeriveCount(graph, pos);
        return {end.x, end.y, end_heading, 0.0};
    }

    return {-1, -1, -1, 0.0};
}
