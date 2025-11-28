
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../../include/graph.h"
#include <math_constants.h>

extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);
__global__ static void __CUDA_KERNEL_count_elements_in_graph(int4 *graph, int width, int height, int type, unsigned int *count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (graph[pos].z == type)
    {
        // int z = pos / width;
        // int x = pos - z * width;
        // printf("%d, %d is in graph, inc count...\n", x, z);
        atomicInc(count, width * height);
    }
}

unsigned int CudaGraph::count(int type)
{
    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_parallelCount->get() = 0;
    __CUDA_KERNEL_count_elements_in_graph<<<numBlocks, THREADS_IN_BLOCK>>>(_graph->getCudaPtr(), _graph->width(), _graph->height(), type, _parallelCount->get());
    cudaDeviceSynchronize();

    return *_parallelCount->get();
}

__global__ static void __CUDA_KERNEL_count_all_elements_in_graph(int4 *graph, int width, int height,  unsigned int *count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (graph[pos].z != GRAPH_TYPE_NULL)
    {
        // int z = pos / width;
        // int x = pos - z * width;
        // printf("%d, %d is in graph, inc count...\n", x, z);
        atomicInc(count, width * height);
    }
}



unsigned int CudaGraph::countAll()
{
    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_parallelCount->get() = 0;
    __CUDA_KERNEL_count_all_elements_in_graph<<<numBlocks, THREADS_IN_BLOCK>>>(_graph->getCudaPtr(), _graph->width(), _graph->height(), _parallelCount->get());
    cudaDeviceSynchronize();

    return *_parallelCount->get();
}

__global__ static void __CUDA_KERNEL_list_elements_in_graph(int4 *graph, int width, int height, int type, int2 *res, unsigned int *currentPos)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z == type)
    {
        int store_pos = atomicInc(currentPos, width * height);
        res[store_pos].x = x;
        res[store_pos].y = z;
    }
}

std::pair<int2 *, int> CudaGraph::__listNodes(int type) {
    unsigned int c = count(type);

    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    int2 *cudaResult = nullptr;
    
    size_t nodeListSize = sizeof(int2) * (c + 1);

    if (!cudaAllocMapped(&cudaResult, nodeListSize))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(nodeListSize) + std::string(" bytes for listing\n");
        throw msg;
    }

    unsigned int *listPos = nullptr;
    if (!cudaAllocMapped(&listPos, sizeof(unsigned int)))
    {
        cudaFreeHost(cudaResult);
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(unsigned int)) + std::string(" bytes for listing\n");
        throw msg;
    }
    *listPos = 0;

    __CUDA_KERNEL_list_elements_in_graph<<<numBlocks, THREADS_IN_BLOCK>>>(_graph->getCudaPtr(), _graph->width(), _graph->height(), type, cudaResult, listPos);
    CUDA(cudaDeviceSynchronize());

    cudaFreeHost(listPos);

    return {cudaResult, c};
}


std::vector<int2> CudaGraph::list()
{
    std::pair<int2 *, int> lst = __listNodes(GRAPH_TYPE_NODE);
    int count = lst.second;
    int2 * cudaResult = lst.first;

    std::vector<int2> res;
    for (int i = 0; i < count; i++)
    {
        res.push_back({cudaResult[i].x, cudaResult[i].y});
    }

    cudaFreeHost(cudaResult);
    
    return res;
}


__global__ static void __CUDA_KERNEL_list_all_elements_in_graph(int4 *graph, int width, int height, int3 *res, unsigned int *currentPos)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_NULL)
    {
        int store_pos = atomicInc(currentPos, width * height);
        res[store_pos].x = x;
        res[store_pos].y = z;
        res[store_pos].z = getTypeCuda(graph, pos);
    }
}


std::pair<int3 *, int> CudaGraph::__listAllNodes() {
    unsigned int c = countAll();

    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    int3 *cudaResult = nullptr;
    
    size_t nodeListSize = sizeof(int3) * (c + 1);

    if (!cudaAllocMapped(&cudaResult, nodeListSize))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(nodeListSize) + std::string(" bytes for listing\n");
        throw msg;
    }

    unsigned int *listPos = nullptr;
    if (!cudaAllocMapped(&listPos, sizeof(unsigned int)))
    {
        cudaFreeHost(cudaResult);
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(unsigned int)) + std::string(" bytes for listing\n");
        throw msg;
    }
    *listPos = 0;

    __CUDA_KERNEL_list_all_elements_in_graph<<<numBlocks, THREADS_IN_BLOCK>>>(_graph->getCudaPtr(), _graph->width(), _graph->height(), cudaResult, listPos);
    CUDA(cudaDeviceSynchronize());


    cudaFreeHost(listPos);

    return {cudaResult, c};
}

std::vector<int3> CudaGraph::listAll()
{
    std::pair<int3 *, int> lst = __listAllNodes();
    int count = lst.second;
    int3 * cudaResult = lst.first;

    std::vector<int3> res;
    for (int i = 0; i < count; i++)
    {
        res.push_back({cudaResult[i].x, cudaResult[i].y, cudaResult[i].z});
    }

    cudaFreeHost(cudaResult);
    
    return res;
}

extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);
__global__ static void __CUDA_KERNEL_check_new_nodes_added(int4 *graph, int width, int height, bool *added)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (getTypeCuda(graph, pos) == GRAPH_TYPE_TEMP)
        *added = true;
}

bool CudaGraph::checkNewNodesAddedOnTreeExpansion()
{
    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_newNodesAdded->get() = false;
    __CUDA_KERNEL_check_new_nodes_added<<<numBlocks, THREADS_IN_BLOCK>>>(_graph->getCudaPtr(), _graph->width(), _graph->height(), _newNodesAdded->get());

    CUDA(cudaDeviceSynchronize());
    
    return *_newNodesAdded->get();
}