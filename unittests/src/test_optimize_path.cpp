#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include "test_utils.h"
#include "../../include/graph.h"
#include <string>
#define PHYS_SIZE 34.641016151377535

// TEST(TestOptimizeGraphs, TestOptimizeNode)
// {
//     CudaGraph g(256, 256);
//     float3 *ptr = createEmptySearchFrame(256, 256);
//     angle maxSteering = angle::deg(40);
//     std::vector<float> costs = {
//                    {0},
//                    {1},
//                    {2},
//                    {3},
//                    {4},
//                    {5}};
//     g.setPhysicalParams(PHYS_SIZE, PHYS_SIZE, maxSteering, 5.412658773);
//     g.setClassCosts(costs);
//     g.setSearchParams({0, 0}, {-1, -1}, {-1, -1});
//     // tstFrame.addArea(130, 90, 140, 70, 28);

//     g.add(128, 128, angle::rad(0.0), -1, -1, 0.0);
//     g.add(127, 121, angle::rad(-0.05952281504869461), 128, 128, 10.0);
//     g.add(127, 116, angle::rad(-0.035784658044576645), 127, 121, 20.0);
//     g.add(128, 105, angle::rad(0.010117189027369022), 127, 116, 30.0);
//     g.add(105, 92, angle::rad(-0.1315600872039795), 128, 105, 40.0);
//     g.add(96, 70, angle::rad(-0.353880912065506), 105, 92, 50.0);
//     g.add(112, 37, angle::rad(-0.10171803086996078), 96, 70, 60.0);
//     g.add(112, 17, angle::rad(-0.022149957716464996), 112, 37, 70.0);
//     g.add(113, 7, angle::rad(0.015074538066983223), 112, 17, 80.0);
//     g.add(114, 0, angle::rad(0.04908384382724762), 113, 7, 90.0);

//     //exportGraph(&g, "test.png");

//     g.addTemporary(128, 10, angle::rad(0.03), 96, 70, 45.0);
//     g.optimizeNode(ptr, 128, 10, 30.0, 1.0, 11);

//     //exportGraph(&g, "test.png");
//     g.acceptDerivedNodes();

//     int2 parent;

//     parent = g.getParent(114, 0);
//     ASSERT_EQ(parent.x, 128);
//     ASSERT_EQ(parent.y, 10);

//     parent = g.getParent(113, 7);
//     ASSERT_EQ(parent.x, 128);
//     ASSERT_EQ(parent.y, 10);

//     parent = g.getParent(128, 10);
//     ASSERT_EQ(parent.x, 96);
//     ASSERT_EQ(parent.y, 70);

//     int j = 1;
// }

int2 last = {-1, -1};
float lastCost = 0;

void addToGraphSeq(CudaGraph &g, int2 next, angle heading)
{
    float cost;
    if (last.x == -1)
    {
        cost = 0;
    }
    else
    {
        int dx = next.x - last.x;
        int dz = next.y - last.y;
        lastCost += 1.5 * sqrtf(dx * dx + dz * dz);
        cost = lastCost;
    }
    g.add(next.x, next.y, heading, last.x, last.y, cost);
    last.x = next.x;
    last.y = next.y;

    printf("added (%d, %d) with cost %f\n", last.x, last.y, cost);
}

// void showPathChanged(std::vector<int2> p1, std::vector<int2> p2)
// {
//     int c1 = p1.size();
//     int c2 = p1.size();

//     if (c1 < c2)
//     {
//         for (int i = 0; i < c1; i++)
//         {
//             if (p1[i].x != p2[i].x || p1[i].y != p2[i].y)
//                 printf("(%d,%d) <> (%d,%d) ", p1[i].x, p1[i].y, p2[i].x, p2[i].y);
//         }
//         for (int i = c1; i < c2; i++)
//         {
//             printf("(+)(%d,%d) ", p2[i].x, p2[i].y);
//         }
//     }
//     printf("\n");
// }

float sumPathTotalCost(std::vector<int2> path, CudaGraph *g)
{
    float cost = 0;
    for (int i = 0; i < path.size(); i++)
    {
        cost += g->getCost(path[i].x, path[i].y);
    }
    return cost;
}

TEST(TestOptimizeGraphs, TestOptimizeGraph)
{
    CudaGraph g(256, 256);
    CudaPtr<float3> ptr = createEmptySearchFrame(256, 256);
    angle maxSteering = angle::deg(40);
    std::vector<float> costs = {
                   {0},
                   {1},
                   {2},
                   {3},
                   {4},
                   {5}};
    g.setPhysicalParams(PHYS_SIZE, PHYS_SIZE, maxSteering, 5.412658773, -1);
    g.setClassCosts(costs);
    g.setSearchParams({0, 0}, {-1, -1}, {-1, -1});

    addToGraphSeq(g, {128, 128}, angle::rad(0.0));
    addToGraphSeq(g, {127, 121}, angle::rad(-0.05952281504869461));
    addToGraphSeq(g, {127, 116}, angle::rad(-0.035784658044576645));
    addToGraphSeq(g, {128, 105}, angle::rad(0.010117189027369022));
    addToGraphSeq(g, {105, 92}, angle::rad(-0.1315600872039795));
    addToGraphSeq(g, {96, 70}, angle::rad(-0.353880912065506));
    addToGraphSeq(g, {112, 37}, angle::rad(-0.10171803086996078));
    addToGraphSeq(g, {112, 17}, angle::rad(-0.022149957716464996));
    addToGraphSeq(g, {113, 7}, angle::rad(0.015074538066983223));
    addToGraphSeq(g, {114, 0}, angle::rad(0.04908384382724762));

    float3 goal = {128.0, 0.0, 0.0};
    CudaPtr<float3> og = createEmptySearchFrame(g.width(), g.height());

    std::vector<int2> plannedPath = get_planned_path(&g, og.get(), angle::rad(0.0), 128, 0, 20.0);
    std::vector<int2> newPlannedPath;

    exportGraph(&g, "test.png", &plannedPath);

    for (int i = 0; i < 100; i++)
    {
        g.optimizeGraph(ptr.get(), {128, 0}, angle::rad(0), 20.0, 1.0);      
        newPlannedPath = get_planned_path(&g, og.get(), angle::rad(0.0), 128, 0, 20.0);
        g.clear();
        for (int j = 0; j < newPlannedPath.size(); j++)
        {
            g.setType(newPlannedPath[j].x, newPlannedPath[j].y, GRAPH_TYPE_NODE);
        }
    }

    newPlannedPath = get_planned_path(&g, og.get(), angle::rad(0.0), 128, 0, 20.0);

    float mean_original_path_cost = sumPathTotalCost(plannedPath, &g) / plannedPath.size();
    
    
    float mean_optimized_path_cost = sumPathTotalCost(newPlannedPath, &g) / newPlannedPath.size();
    //exportGraph(&g, "test.png", &newPlannedPath);

    ASSERT_LE(mean_optimized_path_cost, mean_original_path_cost);   
}
/*
TEST(TestOptimizeGraphs, TestDebugOptimizeGraph)
{
    return;
    CudaGraph g(256, 256);
    float3 *ptr = createEmptySearchFrame(256, 256);
    angle maxSteering = angle::deg(40);
    int costs[] = {{0},
                   {1},
                   {2},
                   {3},
                   {4},
                   {5}};
    g.setPhysicalParams(PHYS_SIZE, PHYS_SIZE, maxSteering, 5.412658773);
    g.setClassCosts(costs, 6);
    g.setSearchParams({0, 0}, {-1, -1}, {-1, -1});
    float3 *og = createEmptySearchFrame(g.width(), g.height());

    for (int i = 0; i < 10; i++)
    {
        std::string s = "../tstlog/graph" + std::to_string(i + 1) + ".dat";
        g.readfromDump(s.c_str());
        std::vector<int2> newPlannedPath = get_planned_path(&g, og, angle::rad(0.0), 128, 0, 30.0);
        exportGraph(&g, "test.png", &newPlannedPath);

        g.optimizeGraph(ptr, angle::rad(0), 30.0, 1.0);

        newPlannedPath = get_planned_path(&g, og, angle::rad(0.0), 128, 0, 30.0);
        exportGraph(&g, "test.png", &newPlannedPath);
    }
} */