

#include <driveless/cpu_parallel_processor.h>
#include "../../include/cuda_graph.h"
#include "atomic_utils.h"
#include <cstdlib>
#include <ctime>
#include <stdatomic.h>
#include <limits>

extern void setupRandomGen(RandState* states, int size, uint64_t seed);

long computePos(int width, int x, int z)
{
    return z * width + x;
}

bool set(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override)
{
    if (override)
    {
        int *addr = &graph[pos].z;
        __atomic_exchange_n(addr, type, __ATOMIC_SEQ_CST);
    }
    else
    {
        int expected = GRAPH_TYPE_NULL;
        bool res = __atomic_compare_exchange_n(&graph[pos].z, &expected, type, false,
                                               __ATOMIC_SEQ_CST, // success memory order
                                               __ATOMIC_SEQ_CST  // failure memory order
        );
        if (res)
        {
            if (graph[pos].z != GRAPH_TYPE_COLLISION &&
                graph[pos].z != GRAPH_TYPE_NODE &&
                graph[pos].z != GRAPH_TYPE_NULL &&
                graph[pos].z != GRAPH_TYPE_PROCESSING &&
                graph[pos].z != GRAPH_TYPE_TEMP &&
                graph[pos].z != GRAPH_TYPE_CONNECT_TO_GOAL)
            {
                printf("set erro on pos = %ld\n", pos);
            }
            return false;
        }
        else
        {
            if (expected != GRAPH_TYPE_NULL)
            {
                if (graph[pos].z != GRAPH_TYPE_COLLISION &&
                    graph[pos].z != GRAPH_TYPE_NODE &&
                    graph[pos].z != GRAPH_TYPE_NULL &&
                    graph[pos].z != GRAPH_TYPE_PROCESSING &&
                    graph[pos].z != GRAPH_TYPE_TEMP &&
                    graph[pos].z != GRAPH_TYPE_CONNECT_TO_GOAL)
                {
                    printf("set erro on pos = %ld\n", pos);
                }

                return false;
            }
        }
    }

    // will return if z is originally not 0.
    graph[pos].x = parent_x;
    graph[pos].y = parent_z;
    graphData[pos].x = heading;
    graphData[pos].y = cost;

    if (graph[pos].z != GRAPH_TYPE_COLLISION &&
        graph[pos].z != GRAPH_TYPE_NODE &&
        graph[pos].z != GRAPH_TYPE_NULL &&
        graph[pos].z != GRAPH_TYPE_PROCESSING &&
        graph[pos].z != GRAPH_TYPE_TEMP &&
        graph[pos].z != GRAPH_TYPE_CONNECT_TO_GOAL)
    {
        printf("erro no set: para pos = %ld\n", pos);
    }

    return true;
}

bool setCollisionCpu(int4 *graph, float4 *graphData, long pos, float heading, int parent_x, int parent_z, float cost)
{

    int expected = GRAPH_TYPE_NODE;
    bool res = __atomic_compare_exchange_n(&graph[pos].z, &expected, GRAPH_TYPE_COLLISION, false,
                                           __ATOMIC_SEQ_CST, // success memory order
                                           __ATOMIC_SEQ_CST  // failure memory order
    );
    if (!res)
        return false;

    graph[pos].x = parent_x;
    graph[pos].y = parent_z;
    graphData[pos].x = heading;
    graphData[pos].y = cost;
    return true;
}

void setParentCpu(int4 *graph, long pos, int parent_x, int parent_z)
{
    graph[pos].x = parent_x;
    graph[pos].y = parent_z;
}

int2 getParentCpu(int4 *graph, long pos)
{
    return {graph[pos].x, graph[pos].y};
}

void setTypeCpu(int4 *graph, long pos, int type)
{
    graph[pos].z = type;
}

int getTypeCpu(int4 *graph, long pos)
{
    return graph[pos].z;
}

void incNodeDeriveCount(int4 *graph, long pos)
{
    graph[pos].w++;
}
void decNodeDeriveCount(int4 *graph, long pos)
{
    graph[pos].w--;
}

void setNodeDeriveCount(int4 *graph, long pos, int count)
{
    graph[pos].w = count;
}

int getNodeDeriveCount(int4 *graph, long pos)
{
    return graph[pos].w;
}

float getHeadingCpu(float4 *graphData, long pos)
{
    return graphData[pos].x;
}

void setHeadingCpu(float4 *graphData, long pos, float heading)
{
    graphData[pos].x = heading;
}

float getCostCpu(float4 *graphData, long pos)
{
    return graphData[pos].y;
}

void setCostCpu(float4 *graphData, long pos, float cost)
{
    graphData[pos].y = cost;
}

float getIntrinsicCost(float4 *graphData, long pos)
{
    return graphData[pos].z;
}

float getIntrinsicCost(float4 *graphData, int width, int x, int z)
{
    long pos = computePos(width, x, z);
    return graphData[pos].z;
}

void setIntrinsicCost(float4 *graphData, long pos, float cost)
{
    graphData[pos].z = cost;
}
void setIntrinsicCost(float4 *graphData, int width, int x, int z, float cost)
{
    long pos = computePos(width, x, z);
    graphData[pos].z = cost;
}

void incIntrinsicCost(float4 *graphData, int width, int x, int z, float cost)
{
    long pos = computePos(width, x, z);
    atomicAddFloat(&graphData[pos].z, cost);
}

__device__ __host__ bool checkInGraphCpu(int4 *graph, long pos)
{
    return graph[pos].z == GRAPH_TYPE_NODE;
}

__device__ __host__ void setDirectCostCpu(float4 *graphData, long pos, float cost)
{
    graphData[pos].w = cost;
}
__device__ __host__ float getDirectCostCpu(float4 *graphData, long pos)
{
    return graphData[pos].w;
}

__device__ __host__ void assertDAGconsistency(int4 *graph, float4 *graphData, int width, int height, long pos)
{
    long curr_pos = pos;
    long i = width * height;
    while (i-- > 0)
    {
        int2 parent = getParentCpu(graph, curr_pos);
        if (parent.x == -1)
            return;
        curr_pos = computePos(width, parent.x, parent.y);
        if (curr_pos == pos)
            break;
    }

    int z = pos / width;
    int x = pos - z * width;

    int2 parent = getParentCpu(graph, pos);
    printf("[CUDA ERROR] DAG is inconsistent on %d, %d, parents %d, %d\n", x, z, parent.x, parent.y);
}

__device__ __host__ double computeHeading(int x1, int z1, int x2, int z2)
{
    double dz = z2 - z1;
    double dx = x2 - x1;

    if (dx == 0 && dz == 0)
        return 0;

    double v1 = 0;
    if (dz != 0)
        v1 = atan2f(-dz, dx);
    else
        v1 = atan2f(0, dx);

    return HALF_PI - v1;
}

float CudaGraph::getDirectCost(int x, int z)
{
    return getDirectCostCpu(_graphData->getPtr(), computePos(_graph->width(), x, z));
}

void CudaGraph::setType(int x, int z, int type)
{
    long pos = computePos(_graph->width(), x, z);
    setTypeCpu(_graph->getPtr(), pos, type);
}

CudaGraph::CudaGraph(int width, int height)
{
    _graph = std::make_shared<Frame<int4>>(width, height);
    _graphData = std::make_unique<Frame<float4>>(width, height);
    _graphCollision = std::make_unique<Frame<float4>>(width, height);
    _graphGoalDirectConnection = std::make_unique<Frame<float3>>(width, height);
    _parallelCount = 1;
    _physicalParams = nullptr;
    _searchSpaceParams = std::make_unique<int[]>(20);
    _searchSpaceParams.get()[FRAME_PARAM_WIDTH] = width;
    _searchSpaceParams.get()[FRAME_PARAM_HEIGHT] = height;
    _searchSpaceParams.get()[FRAME_PARAM_CENTER_X] = TO_INT(width / 2);
    _searchSpaceParams.get()[FRAME_PARAM_CENTER_Z] = TO_INT(height / 2);
    _classCosts = nullptr;
    _randState = std::make_unique<RandState[]>(width * height);
    setupRandomGen(_randState.get(), width * height, 43ULL);

    _bestNodeDirectConnection = std::make_unique<float4[]>(2);
    _bestNodeDirectConnectionCost =  std::numeric_limits<int>::max();
    _region_node_count = nullptr;
    

    _goalReached = false;
    _newNodesAdded = false;
    _nodeCollision = false;
    _ogCoordinateStart = {0, 0, 0};
    __initializeRegionDensity();
    _directOptimPos = -1;



    // default coordinate system start is the middle of the map with heading = 0.0
    setCoordinateStart(_searchSpaceParams.get()[FRAME_PARAM_CENTER_X], _searchSpaceParams.get()[FRAME_PARAM_CENTER_Z]);
}
CudaGraph::~CudaGraph()
{
    __dealocRegionDensity();
}

void CudaGraph::setPhysicalParams(float perceptionWidthSize_m, float perceptionHeightSize_m, angle maxSteeringAngle, float vehicleLength, float max_curvature)
{
    _physicalParams = std::make_unique<double[]>(9);
    _physicalParams.get()[PHYSICAL_PARAMS_RATE_W] = _graph->width() / perceptionWidthSize_m;
    _physicalParams.get()[PHYSICAL_PARAMS_INV_RATE_W] = perceptionWidthSize_m / _graph->width();
    _physicalParams.get()[PHYSICAL_PARAMS_RATE_H] = _graph->height() / perceptionHeightSize_m;
    _physicalParams.get()[PHYSICAL_PARAMS_INV_RATE_H] = perceptionHeightSize_m / _graph->height();
    _physicalParams.get()[PHYSICAL_PARAMS_MAX_STEERING_RAD] = maxSteeringAngle.rad();
    _physicalParams.get()[PHYSICAL_PARAMS_MAX_STEERING_DEG] = maxSteeringAngle.deg();
    _physicalParams.get()[PHYSICAL_PARAMS_LR] = vehicleLength / 2;

    const float t = tanf(maxSteeringAngle.rad());

    if (max_curvature < 0)
        _physicalParams.get()[PHYSICAL_MAX_CURVATURE] = 2 * t / (0.5 * vehicleLength * sqrtf(4 + t));
    else
        _physicalParams.get()[PHYSICAL_MAX_CURVATURE] = max_curvature;
}

void CudaGraph::setSearchParams(std::pair<int, int> minDistance, std::pair<int, int> lowerBound, std::pair<int, int> upperBound)
{
    _searchSpaceParams.get()[FRAME_PARAM_MIN_DIST_X] = TO_INT((float)minDistance.first / 2);
    _searchSpaceParams.get()[FRAME_PARAM_MIN_DIST_Z] = TO_INT((float)minDistance.second / 2);
    _searchSpaceParams.get()[FRAME_PARAM_LOWER_BOUND_X] = lowerBound.first;
    _searchSpaceParams.get()[FRAME_PARAM_LOWER_BOUND_Z] = lowerBound.second;
    _searchSpaceParams.get()[FRAME_PARAM_UPPER_BOUND_X] = upperBound.first;
    _searchSpaceParams.get()[FRAME_PARAM_UPPER_BOUND_Z] = upperBound.second;
}

void CudaGraph::setClassCosts(float *costs, int count)
{
    _classCosts = std::make_unique<float[]>(count);

    auto ptr = _classCosts.get();

    for (int i = 0; i < count; i++)
    {
        ptr[i] = costs[i];
    }
}

void CudaGraph::setClassCosts(std::vector<float> costs)
{
    _classCosts = std::make_unique<float[]>(costs.size());

    auto ptr = _classCosts.get();

    for (int i = 0; i < costs.size(); i++)
    {
        ptr[i] = static_cast<float>(costs[i]);
    }
}

float3 CudaGraph::getCoordinateStart()
{
    return _ogCoordinateStart;
}

void CudaGraph::addStart(int x, int z, angle heading)
{

    add(x, z, heading, -1, -1, 0);
}

void CudaGraph::setCoordinateStart(int x, int z, angle heading)
{
    _ogCoordinateStart.x = static_cast<float>(x);
    _ogCoordinateStart.y = static_cast<float>(z);
    _ogCoordinateStart.z = static_cast<float>(heading.rad());
}
void CudaGraph::setCoordinateStart(int x, int z)
{
    _ogCoordinateStart.x = static_cast<float>(x);
    _ogCoordinateStart.y = static_cast<float>(z);
    _ogCoordinateStart.z = 0.0;
}

void CudaGraph::add(int x, int z, angle heading, int parent_x, int parent_z, float cost)
{
    if (!__checkLimits(x, z))
        return;
    long pos = computePos(_graph->width(), x, z);

    if (parent_x != -1 && parent_z != -1)
        incNodeDeriveCount(_graph->getPtr(), computePos(_graph->width(), parent_x, parent_z));

    set(_graph->getPtr(), _graphData->getPtr(), pos, heading.rad(), parent_x, parent_z, cost, GRAPH_TYPE_NODE, true);
}
void CudaGraph::addTemporary(int x, int z, angle heading, int parent_x, int parent_z, float cost)
{
    if (!__checkLimits(x, z))
        return;
    long pos = computePos(_graph->width(), x, z);
    set(_graph->getPtr(), _graphData->getPtr(), pos, heading.rad(), parent_x, parent_z, cost, GRAPH_TYPE_TEMP, true);
}

bool CudaGraph::__checkLimits(int x, int z)
{
    if (x < 0 || x >= _graph->width())
        return false;
    if (z < 0 || z >= _graph->height())
        return false;

    return true;
}

void CudaGraph::remove(int x, int z)
{
    if (!__checkLimits(x, z))
        return;
    setTypeCpu(_graph->getPtr(), computePos(_graph->width(), x, z), GRAPH_TYPE_NULL);
}

class GraphClearProcessor : public ParallelProcessor
{
private:
    int4 *_graph;
    int _width;
    int _height;
    int _max;

public:
    GraphClearProcessor(int4 *graph, int width, int height, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, height, width),
                                                                                          _graph(graph), _width(width), _height(height)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;
        _graph[pos].z = GRAPH_TYPE_NULL;
    }
};

void CudaGraph::clear()
{
    _directOptimPos = -1;
    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    GraphClearProcessor(_graph->getPtr(), width(), height()).runAndWait();
    _goalReached = false;
    _nodeCollision = false;
}

bool CudaGraph::checkInGraph(int x, int z)
{
    if (!__checkLimits(x, z))
        return false;

    long pos = computePos(_graph->width(), x, z);
    return checkInGraphCpu(_graph->getPtr(), pos);
}

void CudaGraph::setParent(int x, int z, int parent_x, int parent_z)
{
    if (!__checkLimits(x, z))
        return;
    long pos = computePos(_graph->width(), x, z);
    setParentCpu(_graph->getPtr(), pos, parent_x, parent_z);
}

int2 CudaGraph::getParent(int x, int z)
{
    if (!__checkLimits(x, z) || getType(x, z) == GRAPH_TYPE_NULL)
        return {-1, -1};

    long pos = computePos(_graph->width(), x, z);
    return getParentCpu(_graph->getPtr(), pos);
}

angle CudaGraph::getHeading(int x, int z)
{
    long pos = computePos(_graphData->width(), x, z);
    return angle::rad(getHeadingCpu(_graphData->getPtr(), pos));
}

void CudaGraph::setHeading(int x, int z, angle heading)
{
    if (!__checkLimits(x, z))
        return;

    long pos = computePos(_graphData->width(), x, z);
    setHeadingCpu(_graphData->getPtr(), pos, heading.rad());
}

float CudaGraph::getCost(int x, int z)
{
    if (!__checkLimits(x, z))
        return -1;

    long pos = computePos(_graphData->width(), x, z);
    return getCostCpu(_graphData->getPtr(), pos);
}
void CudaGraph::setCost(int x, int z, float cost)
{
    if (!__checkLimits(x, z))
        return;

    long pos = computePos(_graphData->width(), x, z);
    setCostCpu(_graphData->getPtr(), pos, cost);
}

int CudaGraph::getType(int x, int z)
{
    if (!__checkLimits(x, z))
        return -1;

    long pos = computePos(_graph->width(), x, z);
    return getTypeCpu(_graph->getPtr(), pos);
}

void CudaGraph::dumpGraph(const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening file %s\n", filename);
        return;
    }

    int4 *fptr = _graph->getPtr();
    float4 *fptrData = _graphData->getPtr();

    for (int z = 0; z < _graph->height(); z++)
    {
        for (int x = 0; x < _graph->width(); x++)
        {
            long pos = z * _graph->width() + x;
            fprintf(fp, "%d %d %d %d %f %f %f %f\n", fptr[pos].x, fptr[pos].y, fptr[pos].z, fptr[pos].w,
                    fptrData[pos].x, fptrData[pos].y, fptrData[pos].z, fptrData[pos].w);
        }
    }

    fclose(fp);
    printf("Graph dumped to %s\n", filename);
}

void CudaGraph::readfromDump(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Error opening file %s\n", filename);
        return;
    }

    int4 *fptr = _graph->getPtr();
    float4 *fptrData = _graphData->getPtr();

    for (int z = 0; z < _graph->height(); z++)
    {
        for (int x = 0; x < _graph->width(); x++)
        {
            long pos = z * _graph->width() + x;
            fscanf(fp, "%d %d %d %d %f %f %f %f\n", &fptr[pos].x, &fptr[pos].y, &fptr[pos].z, &fptr[pos].w,
                   &fptrData[pos].x, &fptrData[pos].y, &fptrData[pos].z, &fptrData[pos].w);
        }
    }

    fclose(fp);
    printf("Graph read from %s\n", filename);
}

int CudaGraph::getChildCount(int x, int z)
{
    return getNodeDeriveCount(_graph->getPtr(), computePos(_graph->width(), x, z));
}

void CudaGraph::setCollision(int x, int z, int new_parent_x, int new_parent_z, angle new_heading, float new_cost)
{
    long pos = computePos(_graph->width(), x, z);
    int2 currentParent = getParentCpu(_graph->getPtr(), pos);
    long currentParentPos = computePos(_graph->width(), currentParent.x, currentParent.y);

    if (setCollisionCpu(_graph->getPtr(), _graphData->getPtr(), pos, new_heading.rad(), new_parent_x, new_parent_z, new_cost))
    {
        decNodeDeriveCount(_graph->getPtr(), currentParentPos);
    }
}
