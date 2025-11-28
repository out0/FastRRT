#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <driveless/coord_conversion.h>
#include <driveless/world_pose.h>
#include <driveless/cuda_ptr.h>
#include "test_utils.h"

#define PHYS_SIZE 34.641016151377535

CudaGraph *buildTestGraphHermite()
{
    CudaGraph *g = new CudaGraph(256, 256);
    angle maxSteering = angle::deg(40);
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {5}};

    g->setPhysicalParams(256, 256, maxSteering, 5.412658773, -1);
    g->setClassCosts(costs);
    g->setSearchParams({0, 0}, {-1, -1}, {-1, -1});
    return g;
}

TEST(TestGraphCanConnectHermite, StraightConnectionSuccess)
{
    MapPose location(0, 0, 0, angle::deg(0));

    auto graph = buildTestGraphHermite();

    graph->add(128, 128, angle::deg(0.0), -1, -1, 0);

    SearchFrame * ptr = createEmptySearchFramePtr(256, 256);
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {5}};
    ptr->setClassCosts(costs);
    ptr->processDistanceToGoal(128, 0);
    ASSERT_TRUE(graph->canConnectToGoal(ptr, 128, 128, 128, 0, 0.0));
    
    delete ptr;
    delete graph;
}

TEST(TestGraphCanConnectHermite, StraightConnectionFail)
{
    MapPose location(0, 0, 0, angle::deg(0));

    auto graph = buildTestGraphHermite();

    graph->add(128, 128, angle::deg(0.0), -1, -1, 0);

    SearchFrame * ptr = createEmptySearchFramePtr(256, 256);
    std::vector<float> costs = {
        {-1},
        {1},
        {2},
        {3},
        {4},
        {5}};
    ptr->setClassCosts(costs);

    ptr->setValues(128, 0, 0, 0, 0);

    ASSERT_TRUE(ptr->isObstacle(128, 0));

    ptr->processDistanceToGoal(128, 0);
    ASSERT_FALSE(graph->canConnectToGoal(ptr, 128, 128, 128, 0, 0.0));
    
    delete ptr;
    delete graph;
}

TEST(TestGraphCanConnectHermite, NoPreProcessFail)
{
    MapPose location(0, 0, 0, angle::deg(0));

    auto graph = buildTestGraphHermite();

    graph->add(128, 128, angle::deg(0.0), -1, -1, 0);

    SearchFrame * ptr = createEmptySearchFramePtr(256, 256);
    std::vector<float> costs = {
        {-1},
        {1},
        {2},
        {3},
        {4},
        {5}};
    ptr->setClassCosts(costs);

    ptr->setValues(128, 0, 0, 0, 0);

    ASSERT_TRUE(ptr->isObstacle(128, 0));
    ASSERT_FALSE(graph->canConnectToGoal(ptr, 128, 128, 128, 0, 0.0));
    
    delete ptr;
    delete graph;
}
