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


TEST(TestGraphConsistencyCheck, CheckEmptyGraph)
{
    CudaGraph *graph = buildTestGraph();
    ASSERT_TRUE(graph->checkGraphIsConsistent());
}

TEST(TestGraphConsistencyCheck, CheckOneNodeGraph)
{
    CudaGraph *graph = buildTestGraph();
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    ASSERT_TRUE(graph->checkGraphIsConsistent());
}


TEST(TestGraphConsistencyCheck, CheckTwoNodesGraph)
{
    CudaGraph *graph = buildTestGraph();
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(128, 78, angle::rad(0), 128, 128, 0);
    ASSERT_TRUE(graph->checkGraphIsConsistent());
}


TEST(TestGraphConsistencyCheck, CheckCircle)
{
    CudaGraph *graph = buildTestGraph();
    //
    //  (78, 90) ---> (128, 78) ----> (128, 48)
    //   /\                               |
    //   |--------------------------------+
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(128, 48, angle::rad(0), 78, 90, 0);
    graph->add(78, 90, angle::rad(0), 128, 78, 0);
    graph->add(128, 78, angle::rad(0), 128, 48, 0);   
    ASSERT_FALSE(graph->checkGraphIsConsistent());
}

TEST(TestGraphConsistencyCheck, CheckTwoNodeCircle)
{
    CudaGraph *graph = buildTestGraph();
    //
    //  (78, 90) ---> (128, 78)
    //   /\             |
    //   |--------------+
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(78, 90, angle::rad(0), 128, 78, 0);
    graph->add(128, 78, angle::rad(0), 78, 90, 0);   
    ASSERT_FALSE(graph->checkGraphIsConsistent());
}


TEST(TestGraphConsistencyCheck, CheckPointSelf)
{
    CudaGraph *graph = buildTestGraph();
    //
    //  (78, 90) ---> (128, 78)
    //   /\             |
    //   |--------------+
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(78, 90, angle::rad(0), 78, 90, 0);
    ASSERT_FALSE(graph->checkGraphIsConsistent());
}

TEST(TestGraphConsistencyCheck, RegularDAGWithBranches)
{
    CudaGraph *graph = buildTestGraph();
    angle p = angle::rad(0);

    graph->add(128, 128, p, -1, -1, 0);
    /**/ graph->add(100, 100, p, 128, 128, 0);
    /**/ graph->add(135, 100, p, 128, 128, 0);
    /******/ graph->add(110, 70, p, 135, 100, 0);
    /******/ graph->add(145, 70, p, 135, 100, 0);
    ASSERT_TRUE(graph->checkGraphIsConsistent());
}