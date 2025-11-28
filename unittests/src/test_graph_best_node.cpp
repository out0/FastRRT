#include <driveless/search_frame.h>
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include "test_utils.h"

#define PHYS_SIZE 34.641016151377535

TEST(TestGraphBestNode, TestGoalReached)
{
    CudaGraph g(256, 256);
    CudaPtr<float3> ptr = createEmptySearchFrame(256, 256);
    std::vector<float> costs = {
                   {0},
                   {1},
                   {2},
                   {3},
                   {4},
                   {5}};
    angle maxSteering = angle::deg(40);
    g.setPhysicalParams(PHYS_SIZE, PHYS_SIZE, maxSteering, 5.412658773, -1);
    g.setClassCosts(costs);
    g.setSearchParams({0, 0}, {-1, -1}, {-1, -1});

    ASSERT_FALSE(g.checkGoalReached(ptr.get(), {128, 0}, angle::deg(10), 5.0, 1));

    g.add(128, 128, angle::deg(0), -1, -1, 0);
    g.add(128, 3, angle::deg(10), 128, 128, 10);

    ASSERT_TRUE(g.checkGoalReached(ptr.get(), {128, 0}, angle::deg(10), 5.0, 1));
}

TEST(TestGraphBestNode, TestBestNode)
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

    g.add(128, 128, angle::deg(0), -1, -1, 0);
    g.add(128, 3, angle::deg(0), 128, 128, 10);
    g.add(108, 3, angle::deg(0), 128, 128, 10);
    g.add(158, 3, angle::deg(0), 128, 128, 10);

    int2 p = g.findBestNode(ptr.get(), angle::deg(0), 50.0, 128, 0, 1);

    ASSERT_EQ(p.x, 128);
    ASSERT_EQ(p.y, 3);
}
 