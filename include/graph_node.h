#pragma once

#ifndef __GRAPH_NODE_DRIVELESS_H
#define __GRAPH_NODE_DRIVELESS_H

class GraphNode
{
public:
    int x;
    int z;
    float heading_rad{};
    int nodeType;
    int parent_x{};
    int parent_z{};
    float connectToEndCost{};
    float cost{};

    GraphNode(int x, int z, int type) : x(x), z(z), nodeType(type) {}
    GraphNode() : x(0), z(0), heading_rad(0.0f), nodeType(0), parent_x(0), parent_z(0), connectToEndCost(0.0f), cost(0.0f) {}
};

#define GRAPH_TYPE_NULL 0
#define GRAPH_TYPE_NODE 1
#define GRAPH_TYPE_TEMP 2
#define GRAPH_TYPE_PROCESSING 3
#define GRAPH_TYPE_COLLISION 4
#define GRAPH_TYPE_CONNECT_TO_GOAL 5

#define THREADS_IN_BLOCK 256

typedef float3 pose;

#endif