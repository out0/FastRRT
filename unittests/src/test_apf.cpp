#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include "test_utils.h"

#define PHYS_SIZE 34.641016151377535

TEST(TestSimpleAPF, TestAPF)
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
    g.add(128, 230, angle::rad(0.0), -1, -1, 0);

    CudaPtr<float3> og = createEmptySearchFrame(256, 256);
    og.get()[128 * 256 + 128].x = 2; // single obstacle in 128,128
    g.computeRepulsiveFieldAPF(og.get(), 2.0, 5);
    g.computeAttractiveFieldAPF(og.get(), 3.0, {128, 0});
}