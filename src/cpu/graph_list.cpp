
#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include "../../include/cuda_graph.h"

extern __device__ __host__ int getTypeCpu(int4 *graph, long pos);

class CountElementsInGraphProcessor : public ParallelProcessor
{

private:
    int4 *_graph;
    int _width;
    int _height;
    int _type;
    std::atomic<int> _count{0};
    int _max;

public:
    CountElementsInGraphProcessor(int4 *graph, int width, int height, int type, int numThreadHandlers = 12) //
        : ParallelProcessor(numThreadHandlers, height, width),                                              //
          _graph(graph), _width(width), _height(height),                                                    //
          _type(type)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        if (_graph[pos].z == _type)
        {
            _count.fetch_add(1);
        }
    }

    unsigned int count()
    {
        return _count;
    }
};

unsigned int CudaGraph::count(int type)
{
    auto p = new CountElementsInGraphProcessor(_graph->getPtr(), _graph->width(), _graph->height(), type);
    p->runAndWait();
    int res = p->count();
    delete p;
    return res;
}

class CountAllElementsInGraphProcessor : public ParallelProcessor
{

private:
    int4 *_graph;
    int _width;
    int _height;
    std::atomic<int> _count{0};
    int _max;

public:
    CountAllElementsInGraphProcessor(int4 *graph, int width, int height, int numThreadHandlers = 12) //
        : ParallelProcessor(numThreadHandlers, height, width),                                       //
          _graph(graph), _width(width), _height(height)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        if (_graph[pos].z != GRAPH_TYPE_NULL)
        {
            // int z = pos / width;
            // int x = pos - z * width;
            // printf("%d, %d is in graph, inc count...\n", x, z);
            _count.fetch_add(1);
        }
    }

    unsigned int count()
    {
        return _count;
    }
};

unsigned int CudaGraph::countAll()
{
    auto p = new CountAllElementsInGraphProcessor(_graph->getPtr(), _graph->width(), _graph->height());
    p->runAndWait();
    int res = p->count();
    delete p;
    return res;
}

class ListElementsInGraphProcess : public ParallelProcessor
{

private:
    int4 *_graph;
    int _width;
    int _height;
    int _type;
    int _max;
    int2 *_res;
    std::atomic<int> currentPos{0};

public:
    ListElementsInGraphProcess(int4 *graph, int width, int height,                                                                                                   //
                               int type, int2 *res, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, height, width), _graph(graph), _width(width), //
                                                                                  _height(height), _type(type), _res(res)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int z = pos / _width;
        int x = pos - z * _width;

        if (getTypeCpu(_graph, pos) == _type)
        {
            int store_pos = currentPos.fetch_add(1);
            _res[store_pos].x = x;
            _res[store_pos].y = z;
        }
    }
};

std::pair<int2 *, int> CudaGraph::__listNodes(int type)
{
    unsigned int c = count(type);

    int2 *res = new int2[c + 1];
    ListElementsInGraphProcess(_graph->getPtr(), _graph->width(), _graph->height(), type, res).runAndWait();
    return {res, c};
}

std::vector<int2> CudaGraph::list()
{
    std::pair<int2 *, int> lst = __listNodes(GRAPH_TYPE_NODE);
    int count = lst.second;
    int2 *result = lst.first;

    std::vector<int2> res;
    for (int i = 0; i < count; i++)
    {
        res.push_back({result[i].x, result[i].y});
    }

    delete[] result;
    return res;
}

class ListAllElementsInGraphProcess : public ParallelProcessor
{

private:
    int4 *_graph;
    int _width;
    int _height;
    int _max;
    int3 *_res;
    std::atomic<int> currentPos{0};

public:
    ListAllElementsInGraphProcess(int4 *graph, int width, int height,                                                                                                   //
                                  int3 *res, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, height, width), _graph(graph), _width(width), //
                                                                                     _height(height), _res(res)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int z = pos / _width;
        int x = pos - z * _width;

        if (getTypeCpu(_graph, pos) != GRAPH_TYPE_NULL)
        {
            int store_pos = currentPos.fetch_add(1);
            _res[store_pos].x = x;
            _res[store_pos].y = z;
            _res[store_pos].z = getTypeCpu(_graph, pos);
        }
    }
};

std::pair<int3 *, int> CudaGraph::__listAllNodes()
{
    unsigned int c = countAll();

    int3 *res = new int3[c + 1];
    ListAllElementsInGraphProcess(_graph->getPtr(), _graph->width(), _graph->height(), res).runAndWait();

    return {res, c};
}

std::vector<int3> CudaGraph::listAll()
{
    std::pair<int3 *, int> lst = __listAllNodes();
    int count = lst.second;
    int3 *result = lst.first;

    std::vector<int3> res;
    for (int i = 0; i < count; i++)
    {
        res.push_back({result[i].x, result[i].y, result[i].z});
    }

    delete []result;
    return res;
}



class CheckNewNodesAddedProcess : public ParallelProcessor
{

private:
    int4 *_graph;
    int _width;
    int _height;
    int _max;
    bool _added;

public:
    CheckNewNodesAddedProcess(int4 *graph, int width, int height, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, height, width), _graph(graph), _width(width), //
                                                                                     _height(height)
    {
        _max = width * height;
        _added = false;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        if (getTypeCpu(_graph, pos) == GRAPH_TYPE_TEMP)
        {
           _added = true;
        }
    }

    bool newNodesAdded() {
        return _added;
    }
};


bool CudaGraph::checkNewNodesAddedOnTreeExpansion()
{
    int size = _graph->width() * _graph->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    auto p = new CheckNewNodesAddedProcess(_graph->getPtr(), _graph->width(), _graph->height());
    p->runAndWait();
    bool res = p->newNodesAdded();
    delete p;
    return res;
}