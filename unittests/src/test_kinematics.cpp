#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include "test_utils.h"

extern double2 convert_waypoint_to_map_pose(float3 *ogStart, int2 coord);
extern int2 convert_map_pose_to_waypoint(float3 *ogStart, double2 coord);

#define PHYS_SIZE 34.641016151377535

TEST(TestKinematics, ConvertMapPoseWaypointOrigin)
{

    float3 *ogStart = new float3{128.0, 128.0, 0.0};
    int2 coord = {128, 128};

    double2 mapCoord = convert_waypoint_to_map_pose(ogStart, coord);

    ASSERT_EQ(0, mapCoord.x);
    ASSERT_EQ(0, mapCoord.y);

    int2 waypoint = convert_map_pose_to_waypoint(ogStart, mapCoord);

    ASSERT_EQ(128, waypoint.x);
    ASSERT_EQ(128, waypoint.y);

    delete ogStart;
}

TEST(TestKinematics, ConvertMapPoseWaypointNonOrigin)
{
    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;
    float inv_rw = PHYS_SIZE / 256;
    float inv_rh = PHYS_SIZE / 256;

    float3 *ogStart = new float3{128.0, 128.0, 0.0};
    int2 coord = {108, 108};

    double2 mapCoord = convert_waypoint_to_map_pose(ogStart, coord);

    int2 waypoint = convert_map_pose_to_waypoint(ogStart, mapCoord);

    ASSERT_EQ(108, waypoint.x);
    ASSERT_EQ(108, waypoint.y);
    delete ogStart;
}

TEST(TestKinematics, TestCheckKinematicPath)
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

    g.add(128, 128, angle::rad(0), -1, -1, 0);

    int2 node = g.derivateNode(ptr.get(), angle::deg(20), 50, 2, 128, 128);

    ASSERT_TRUE(g.checkFeasibleConnection(ptr.get(), {128, 128}, node, 2));
}

TEST(TestKinematics, TestCheckKinematicPathShapeZeroHeading)
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

    g.add(128, 128, angle::rad(0), -1, -1, 0);

    int2 node = g.derivateNode(ptr.get(), angle::deg(20), 50, 2, 128, 128);

    ASSERT_TRUE(g.checkFeasibleConnection(ptr.get(), {128, 128}, node, 2));
}