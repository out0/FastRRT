#include <driveless/math_utils.h>
#include <driveless/cuda_params.h>
#include "../../include/cuda_graph.h"


extern __device__ __host__ int getNodeDeriveCount(int4 *graph, long pos);

__device__ __host__ int computeDensityPos(int density_width, int x, int z)
{
    int density_x = TO_INT(x / BLOCK_SIZE);
    int density_z = TO_INT(z / BLOCK_SIZE);
    return (density_z * density_width + density_x);
}


__device__ __host__ bool checkCanExpand(int4 *graph, unsigned int *region_count, int *params, float node_mean, int pos, int x, int z, bool controlExpansion)
{
    if (controlExpansion)
    {
        return getNodeDeriveCount(graph, pos) == 0;
    }

    const int densityPos = computeDensityPos(params[FRAME_DENSITY_WIDTH], x, z);
    // return getNodeDeriveCount(graph, pos) < 3 && (region_count[densityPos] <= 0.5 * BLOCK_SIZE);
    return region_count[densityPos] <= 0.5 * BLOCK_SIZE;
}