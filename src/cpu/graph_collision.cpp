#include "../../include/cuda_graph.h"
#include <driveless/cpu_parallel_processor.h>

extern int2 getParentCpu(int4 *graph, long pos);
extern long computePos(int width, int x, int z);
extern void setTypeCpu(int4 *graph, long pos, int type);
extern int getTypeCpu(int4 *graph, long pos);
extern void incNodeDeriveCount(int4 *graph, long pos);
extern void decNodeDeriveCount(int4 *graph, long pos);
extern void setNodeDeriveCount(int4 *graph, long pos, int count);

class EraseTreesGraphCollisionProcess : public ParallelProcessor
{
private:
    int4 *_graph;
    int *_params;
    int _numNodesInGraph;
    int _max;

public:
    EraseTreesGraphCollisionProcess(int4 *graph, int *params,                          //
                                    int numNodesInGraph, int numThreadHandlers = 12) : //
                                                                                       ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]),
                                                                                       _graph(graph), _params(params), _numNodesInGraph(numNodesInGraph)
    {
        _max = params[FRAME_PARAM_HEIGHT] * params[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {

        if (pos >= _max)
            return;

        int width = _params[FRAME_PARAM_WIDTH];
        int height = _params[FRAME_PARAM_HEIGHT];

        int ptype = getTypeCpu(_graph, pos);

        // printf ("%d\n", ptype);

        if (ptype == GRAPH_TYPE_NULL)
            return;

        // int z = pos / width;
        // int x = pos - z * width;
        int curr = pos;

        int i = _numNodesInGraph + 4;

        while (i-- > 0)
        {
            int2 parent = getParentCpu(_graph, curr);

            if (parent.x == -1 && parent.y == -1)
                return;

            long next = computePos(width, parent.x, parent.y);

            int typeNext = getTypeCpu(_graph, next);

            if (typeNext == GRAPH_TYPE_COLLISION || typeNext == GRAPH_TYPE_NULL)
            {
                // printf("[collision] found collision for %d, %d in node %d, %d\n", x, z, parent.x, parent.y);
                setTypeCpu(_graph, pos, GRAPH_TYPE_NULL);
                return;
            }

            curr = next;
        }

        if (i == 0)
        {
            // cyclic ref.
            // printf("%d, %d is in cyclic ref\n", x, z);
            setTypeCpu(_graph, pos, GRAPH_TYPE_NULL);
        }
    }
};

class SetNodesGraphCollisionProcess : public ParallelProcessor
{
private:
    int4 *_graph;
    int *_params;
    int _max;

public:
    SetNodesGraphCollisionProcess(int4 *graph, int *params,     //
                                  int numThreadHandlers = 12) : //
                                                                ParallelProcessor(numThreadHandlers, params[FRAME_PARAM_HEIGHT], params[FRAME_PARAM_WIDTH]),
                                                                _graph(graph), _params(params)
    {
        _max = params[FRAME_PARAM_HEIGHT] * params[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {

        if (pos >= _max)
            return;
        int width = _params[FRAME_PARAM_WIDTH];
        int height = _params[FRAME_PARAM_HEIGHT];

        if (pos >= width * height)
            return;

        if (getTypeCpu(_graph, pos) == GRAPH_TYPE_COLLISION)
        {
            setTypeCpu(_graph, pos, GRAPH_TYPE_NODE);
            setNodeDeriveCount(_graph, pos, 0);

            int2 parent = getParentCpu(_graph, pos);
            long pos_parent = computePos(width, parent.x, parent.y);
            incNodeDeriveCount(_graph, pos_parent);
        }
    }
};

void CudaGraph::solveCollisions()
{
    int numNodesInGraph = count();

    if (numNodesInGraph <= 2)
        return;

    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    EraseTreesGraphCollisionProcess(_graph->getPtr(), _searchSpaceParams.get(), numNodesInGraph).runAndWait();
    SetNodesGraphCollisionProcess(_graph->getPtr(), _searchSpaceParams.get()).runAndWait();

    _nodeCollision = false;
}