#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include "../../include/cuda_graph.h"


__global__ static void __CUDA_KERNEL_clear(int4 *graph, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    /*
    Clear only sets the type because when we shrink the graph, we want it to preserve the
    original connections, because the shrink simply clears the graph and reset the nodes
    in the path to GRAPH_TYPE_NODE. Thats why the clear must not interfere with x, y values
    (parent values)
    */
    graph[pos].z = GRAPH_TYPE_NULL;
}

void CudaGraph::clear()
{
    _directOptimPos = -1;
    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_clear<<<numBlocks, THREADS_IN_BLOCK>>>(_graph->getPtr(), width(), height());

    CUDA(cudaDeviceSynchronize());
    *_goalReached->get() = false;
    *_nodeCollision->get() = false;
}