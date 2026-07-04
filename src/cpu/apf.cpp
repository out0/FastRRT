
#include "../../include/cuda_graph.h"
#include <driveless/cpu_parallel_processor.h>

#define FORCE_RANGE 5

extern void incIntrinsicCost(float4 *graphData, int width, int x, int z, float cost);
extern long computePos(int width, int x, int z);
extern float getIntrinsicCostCuda(float4 *graphData, long pos);
extern void setIntrinsicCostCuda(float4 *graphData, long pos, float cost);

inline bool in_range(int width, int height, int x, int z)
{
    return x >= 0 && x < width && z >= 0 && z < height;
}

inline bool check_is_obstacle(float3 *og, float *classCosts, int width, int height, int x, int z)
{
    if (!in_range(width, height, x, z))
        return true;
    long pos = computePos(width, x, z);
    return classCosts[TO_INT(og[pos].x)] < 0;
}

class RepulsiveForceProcess : public ParallelProcessor
{
    float3 *_og;
    float4 *_graphData;
    float *_classCosts;
    int *_params;
    int _width;
    int _height;
    float _Kr_half;
    int _radius;
    int _max;

public:
    RepulsiveForceProcess(float3 *og, float4 *graphData,
                          float *classCosts, int *params,
                          int width, int height,
                          float Kr_half, int radius,
                          int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, height, width),
                                                        _og(og), _graphData(graphData), _classCosts(classCosts), _params(params),
                                                        _width(width), _height(height), _Kr_half(Kr_half), _radius(radius)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int z = pos / _width;
        int x = pos - z * _width;

        int lower_bound_ego_x = _params[FRAME_PARAM_LOWER_BOUND_X];
        int lower_bound_ego_z = _params[FRAME_PARAM_LOWER_BOUND_Z];
        int upper_bound_ego_x = _params[FRAME_PARAM_UPPER_BOUND_X];
        int upper_bound_ego_z = _params[FRAME_PARAM_UPPER_BOUND_Z];

        if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
            return;

        // if (x == 46 && z == 46) {
        //     printf ("(%d, %d) classCost = %f\n", x, z);
        // }

        float c = _classCosts[TO_INT(_og[pos].x)];
        if (c >= 0)
            return; // not an obstacle

        setIntrinsicCostCuda(_graphData, pos, 100 * _Kr_half);

        if (check_is_obstacle(_og, _classCosts, _width, _height, x - 1, z) &&
            check_is_obstacle(_og, _classCosts, _width, _height, x + 1, z) &&
            check_is_obstacle(_og, _classCosts, _width, _height, x, z - 1) &&
            check_is_obstacle(_og, _classCosts, _width, _height, x, z + 1))
            return;

        for (int h = z - _radius; h <= z + _radius; h++)
        {
            if (h < 0)
                continue;

            if (h >= _height)
                break;

            int init = x - _radius;
            float p0 = (float)_radius;

            for (int w = init; w <= x + _radius; w++)
            {
                if (w < 0)
                    continue;

                if (w >= _width)
                    break;

                if (w >= lower_bound_ego_x && w <= upper_bound_ego_x && h >= upper_bound_ego_z && h <= lower_bound_ego_z)
                    continue;

                if (w == x && h == z)
                    continue;

                float p = sqrtf((w - x) * (w - x) + (h - z) * (h - z));
                if (p > p0)
                    continue;

                float f = 1 / p - 1 / p0;
                float cost = _Kr_half * f * f;
                incIntrinsicCost(_graphData, _width, w, h, cost);
                // printf("(%d, %d): r = %f, cost =  %f, 1/p = %f, 1/p0 = %f, f = %f\n", w, h, r, cost, 1/p, 1/p0, f);

                // printf("(%d, %d) cost = %f\n", w, h, cost);
            }
        }
    }
};

class AttractiveForceProcess : public ParallelProcessor
{
    float3 *_og;
    float4 *_graphData;
    float *_classCosts;
    int _width;
    int _height;
    float _Ka_half;
    int _goal_x;
    int _goal_z;
    int _max;

public:
    AttractiveForceProcess(float3 *og, float4 *graphData,
                           float *classCosts,
                           int width, int height,
                           float Ka_half, int goal_x,
                           int goal_z,
                           int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, height, width),
                                                         _og(og), _graphData(graphData), _classCosts(classCosts),
                                                         _width(width), _height(height), _Ka_half(Ka_half), _goal_x(goal_x),
                                                         _goal_z(goal_z)

    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int z = pos / _width;
        int x = pos - z * _width;

        float c = _classCosts[TO_INT(_og[pos].x)];
        if (c < 0)
        {
            return; // obstacle
        }

        int dx = _goal_x - x;
        int dz = _goal_z - z;

        float dcost = (float)(dx * dx + dz * dz) * _Ka_half;

        float current_cost = getIntrinsicCostCuda(_graphData, pos);
        setIntrinsicCostCuda(_graphData, pos, current_cost - dcost);
    }
};

void CudaGraph::computeRepulsiveFieldAPF(float3 *og, float Kr, int radius)
{
    RepulsiveForceProcess(og,
                          _graphData->getPtr(),
                          _classCosts.get(),
                          _searchSpaceParams.get(),
                          _graph->width(),
                          _graph->height(),
                          Kr * 0.5,
                          radius)
        .runAndWait();
}
void CudaGraph::computeAttractiveFieldAPF(float3 *og, float Ka, std::pair<int, int> goal)
{
    AttractiveForceProcess(og,
                           _graphData->getPtr(),
                           _classCosts.get(),
                           _graph->width(),
                           _graph->height(),
                           Ka * 0.5,
                           goal.first,
                           goal.second)
        .runAndWait();
}
