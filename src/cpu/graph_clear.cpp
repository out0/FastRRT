#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include "../../include/cuda_graph.h"

class GraphClearProcess : public ParallelProcessor
{

private:
    int4 *_graph;
    std::unique_ptr<int[]> _searchParams;
    int _max;

public:
    GraphClearProcess(int4 *graph,
                      int *searchParams,
                      int numThreadHandlers = 12) : _graph(graph),
                                                    ParallelProcessor(numThreadHandlers,                //
                                                                      searchParams[FRAME_PARAM_HEIGHT], //
                                                                      searchParams[FRAME_PARAM_WIDTH])  //

    {
        _max = searchParams[FRAME_PARAM_HEIGHT] * searchParams[FRAME_PARAM_WIDTH];
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
    GraphClearProcess(_graph->getPtr(), _searchSpaceParams.get()).runAndWait();
    _directOptimPos = -1;
    _goalReached = false;
    _nodeCollision = false;
}