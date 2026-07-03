
#include "../../include/cuda_graph.h"
#include <driveless/cpu_parallel_processor.h>
#include <atomic>

extern int getTypeCpu(int4 *graph, long pos);

class CountElementsInRangeProcessor : public ParallelProcessor
{

private:
    int4 *_graph;
    int _width;
    int _height;
    int _type;
    int _xp;
    int _zp;
    float _radius_sqr;
    unsigned int _count;
    int _max;

public:
    CountElementsInRangeProcessor(int4 *graph, int width, int height, int type,                 //
                                  int xp, int zp, float radius_sqr, int numThreadHandlers = 12) //
        : ParallelProcessor(numThreadHandlers, height, width),                                  //
          _graph(graph), _width(width), _height(height),                                        //
          _type(type), _xp(xp), _zp(zp),
          _radius_sqr(radius_sqr), _count(0)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int z = pos / _width;
        int x = pos - z * _width;

        if (x == _xp && z == _zp)
            return;

        float dx = x - _xp;
        float dz = z - _zp;

        if (dx * dx + dz * dz > _radius_sqr)
            return;

        if (getTypeCpu(_graph, pos) == _type)
        {
            __atomic_fetch_add(&_count, 1u, __ATOMIC_SEQ_CST);
        }
    }

    unsigned int count()
    {
        return _count;
    }
};

unsigned int CudaGraph::__countInRange(int xp, int zp, float radius_sqr)
{
    auto proc = new CountElementsInRangeProcessor(_graph->getPtr(), width(), height(), GRAPH_TYPE_NODE,
                                                  xp, zp, radius_sqr);

    proc->runAndWait();
    return proc->count();
}

class ListElementsInRangeProcess : public ParallelProcessor
{

private:
    int4 *_graph;
    int _width;
    int _height;
    int _type;
    int _xp;
    int _zp;
    int _max;
    float _radius_sqr;
    int2 *_res;
    std::atomic<int> currentPos{0};

public:
    ListElementsInRangeProcess(int4 *graph, int width, int height, int type, int xp,                                                                                            //
                               int zp, float radius_sqr, int2 *res, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, height, width), _graph(graph), _width(width), //
                                                                                             _height(height), _type(type), _xp(xp), _zp(zp), _radius_sqr(radius_sqr),           //
                                                                                             _res(res)
    {
        _max = width * height;
    }

    void handler(int pos) override
    {
        if (pos >= _max)
            return;

        int z = pos / _width;
        int x = pos - z * _width;

        if (x == _xp && z == _zp)
            return;

        float dx = x - _xp;
        float dz = z - _zp;
        if (dx * dx + dz * dz > _radius_sqr)
            return;

        if (getTypeCpu(_graph, pos) == _type)
        {
            int store_pos = currentPos.fetch_add(1);
            _res[store_pos].x = x;
            _res[store_pos].y = z;
        }
    }
};

std::pair<int2 *, int> CudaGraph::__listNodesInRange(int type, int x, int z, float radius_sqr)
{
    unsigned int c = __countInRange(x, z, radius_sqr);

    int size = _graph->width() * _graph->height();

    int2 * res = new int2[c + 1];

    ListElementsInRangeProcess(_graph->getPtr(),
        _graph->width(),
        _graph->height(),
        type,
        x,
        z,
        radius_sqr,
        res).runAndWait();

    return {res, c};
}

std::vector<int2> CudaGraph::listInRange(int x, int z, float radius)
{
    std::pair<int2 *, int> lst = __listNodesInRange(GRAPH_TYPE_NODE, x, z, radius * radius);
    int count = lst.second;
    int2 *result = lst.first;

    std::vector<int2> res;
    for (int i = 0; i < count; i++)
    {
        res.push_back({result[i].x, result[i].y});
    }

    delete []result;
    return res;
}