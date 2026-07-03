
#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include "../../include/cuda_graph.h"

extern int2 getParentCpu(int4 *graph, long pos);
extern float getCostCpu(float4 *graphData, long pos);
extern long computePos(int width, int x, int z);
extern float getHeadingCpu(float4 *graphData, long pos);
extern float getFrameCostCpu(float3 *frame, float *classCost, long pos) ;


float computeCost(float3 *frame, int4 *graph, float4 *graphData, double *physicalParams, float *classCosts, int width, float goalHeading_rad, long nodePos, double distToParent) {
    int2 parent = getParentCpu(graph, nodePos);
    float parentCost = getCostCpu(graphData, computePos(width, parent.x, parent.y));
    float heading_error_perc = abs(goalHeading_rad - getHeadingCpu(graphData, nodePos)) / physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];
    return (getFrameCostCpu(frame, classCosts, nodePos) + distToParent) * (1 + heading_error_perc) + parentCost;
}