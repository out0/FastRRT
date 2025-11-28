
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../include/graph.h"

inline bool sameNode(int2 a, int2 b)
{
    return a.x == b.x && a.y == b.y;
}

void CudaGraph::__printInconsistentChain(int3 n, int maxLoop)
{
    int2 p;
    printf("[GRAPH check] Inconsistent DAG from %d, %d: (%d, %d)", n.x, n.y, n.x, n.y);
    p.x = n.x;
    p.y = n.y;
    for (int i = 0; i < maxLoop; i++)
    {
        int2 parent = getParent(p.x, p.y);
        if (parent.x == -1)
            return;
        if (getType(parent.x, parent.y) == GRAPH_TYPE_COLLISION)
        {
            printf("->(%d, %d) [C]", parent.x, parent.y);
        }
        else
            printf("->(%d, %d)", parent.x, parent.y);
        p.x = parent.x;
        p.y = parent.y;
    }
    printf("\n");
}

bool CudaGraph::checkGraphIsConsistent(bool print_inconsistency)
{
    int maxLoop = count(GRAPH_TYPE_NODE) + 4;
    std::vector<int3> nodes = listAll();

    int2 p;

    for (auto n : nodes)
    {
        p.x = n.x;
        p.y = n.y;
        int i = maxLoop;
        while (i-- > 0)
        {
            int2 parent = getParent(p.x, p.y);
            if (sameNode(p, parent))
            {
                if (print_inconsistency)
                    __printInconsistentChain(n, maxLoop);
                return false;
            }
            if (parent.x == -1)
                break;

            int parentType = getType(parent.x, parent.y);

            if (parentType != GRAPH_TYPE_NODE && parentType != GRAPH_TYPE_COLLISION && parentType != GRAPH_TYPE_CONNECT_TO_GOAL)
            {
                if (print_inconsistency)
                    printf("[GRAPH check] %d, %d is connected to the node (%d, %d) which is not a node: %d\n", p.x, p.y, parent.x, parent.y, getType(parent.x, parent.y));
                return false;
            }
            p.x = parent.x;
            p.y = parent.y;
        }
        if (i <= 0)
        {
            if (print_inconsistency)
                __printInconsistentChain(n, maxLoop);
            return false;
        }
    }
    return true;
}