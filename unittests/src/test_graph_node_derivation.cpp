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

TEST(TestGraphNodeDerivation, StraightDerivationInXcoord)
{
    MapPose location(0, 0, 0, angle::deg(0));

    auto graph = buildTestGraph();

    graph->add(128, 128, angle::deg(0.0), -1, -1, 0);

    const int pathSize = 49;
    const float velocity = 1.0;

    CudaPtr<float3> ptr = createEmptySearchFrame(256, 256);

    graph->derivateNode(ptr.get(), angle::deg(0), pathSize, velocity, 128, 128);
    graph->acceptDerivedNodes({0, 0}, 0.0);

    // ensure that the end-node is 128, 78 with heading 0.0 deg
    if (!graph->checkInGraph(128, 78))
        FAIL();

    ASSERT_EQ(graph->getHeading(128, 78), angle::deg(0.0));

    ///
    /// Waypoint result is independent from map origin, because it is going to be converted back after.
    /// As an example, the 128, 78 is converted here to (50, 0, 0). Check the next test, it is converted to (0, 50, 0)
    /// because map heading is +90 degrees
    ///

    WorldPose wp(angle::deg(0), angle::deg(0), 0, angle::deg(0));
    CoordinateConverter conv(wp, 256, 256, 256, 256);

    Waypoint pose(128, 78, angle::deg(0));
    auto map_point = conv.convert(location, pose);

    ASSERT_DEQ(map_point.x(), 50);
    ASSERT_DEQ(map_point.y(), 0);
    ASSERT_DEQ(map_point.z(), 0);
    ASSERT_EQ(map_point.heading(), angle::deg(0));

    delete graph;
}
TEST(TestGraphNodeDerivation, StraightDerivationInYcoord)
{
    MapPose location(0, 0, 0, angle::deg(90));

    auto graph = buildTestGraph();

    graph->add(128, 128, angle::deg(0.0), -1, -1, 0);

    const int pathSize = 49;
    const float velocity = 1.0;

    CudaPtr<float3> ptr = createEmptySearchFrame(256, 256);

    graph->derivateNode(ptr.get(), angle::deg(0), pathSize, velocity, 128, 128);
    graph->acceptDerivedNodes({0, 0}, 0.0);

    // ensure that the end-node is 128, 78 with heading 0.0 deg
    if (!graph->checkInGraph(128, 78))
        FAIL();

    ASSERT_EQ(graph->getHeading(128, 78), angle::deg(0.0));
    delete graph;

    WorldPose wp(angle::deg(0), angle::deg(0), 0, angle::deg(0));
    CoordinateConverter conv(wp, 256, 256, 256, 256);

    Waypoint pose(128, 78, angle::deg(0));
    auto map_point = conv.convert(location, pose);

    ASSERT_DEQ(map_point.x(), 0);
    ASSERT_DEQ(map_point.y(), 50);
    ASSERT_DEQ(map_point.z(), 0);
    ASSERT_EQ(map_point.heading(), angle::deg(90));
}

TEST(TestGraphNodeDerivation, StraightDerivationWithDelocatedLocalPosition)
{
    MapPose location(0, 0, 0, angle::deg(90));

    auto graph = buildTestGraph();

    graph->add(50, 200, angle::deg(0.0), -1, -1, 0);

    const int pathSize = 49;
    const float velocity = 1.0;

    CudaPtr<float3> ptr = createEmptySearchFrame(256, 256);

    graph->derivateNode(ptr.get(), angle::deg(0), pathSize, velocity, 50, 200);
    graph->acceptDerivedNodes({0, 0}, 0.0);

    // ensure that the end-node is 128, 78 with heading 0.0 deg
    if (!graph->checkInGraph(50, 150))
        FAIL();

    ASSERT_EQ(graph->getHeading(50, 150), angle::deg(0.0));
    delete graph;
}

float3 derive(
    int width,
    int height,
    float width_m,
    float height_m,
    MapPose location,
    Waypoint start,
    float pathSize,
    float steeringAngle,
    float velocity_m_s,
    float lr)
{

    WorldPose wp(angle::rad(0), angle::rad(0), 0.0, angle::rad(0));
    CoordinateConverter conv(
        wp,
        width,
        height,
        width_m,
        height_m);

    conv.setWaypointCoordinateOrigin(128, 128);

    auto mapStart = conv.convert(location, start);

    const float steer = tanf(steeringAngle);
    const float dt = 0.1;
    const float ds = velocity_m_s * dt;
    const float beta = atanf(steer / 2);
    const float heading_increment_factor = ds * cosf(beta) * steer / (2 * lr);

    float x = mapStart.x();
    float y = mapStart.y();
    int maxSize = TO_INT(pathSize) + 1;

    int size = 0;

    int last_x = start.x();
    int last_z = start.z();
    float heading = mapStart.heading().rad();
    float localHeading = start.heading().rad();
    int2 lastp;

    const float parentCost = 0.0;
    float nodeCost = parentCost;

    while (size < maxSize)
    {
        x += ds * cosf(heading + beta);
        y += ds * sinf(heading + beta);
        heading += heading_increment_factor;

        MapPose wpp(x, y, 0, angle::rad(heading));
        Waypoint p = conv.convert(location, wpp);

        lastp.x = p.x();
        lastp.y = p.z();
        localHeading = p.heading().rad();

        if (lastp.x == last_x && lastp.y == last_z)
            continue;

        if (lastp.x < 0 || lastp.x >= width)
            break;

        if (lastp.y < 0 || lastp.y >= height)
            break;

        size += 1;
        nodeCost += 1;

        last_x = lastp.x;
        last_z = lastp.y;
    }

    // printf ("path size: %d\n", size);

    return {(float)last_x, (float)last_z, localHeading};
}

TEST(TestGraphNodeDerivation, CurvyNodeDerivation)
{
    MapPose location(0, 0, 0, angle::deg(0));

    auto graph = buildTestGraph();

    const int pathSize = 15;
    const float velocity = 1.0;

    CudaPtr<float3> ptr = createEmptySearchFrame(256, 256);

    for (int a = -180; a <= 180; a+=1)
    {
        for (int i = -40; i <= 40; i+=18)
        {
            graph->add(100, 200, angle::deg(a), -1, -1, 0);
            graph->derivateNode(ptr.get(), angle::deg(i), pathSize, velocity, 100, 200);
            graph->acceptDerivedNodes({0, 0}, 0.0);

            Waypoint start(100, 200, angle::deg(a));
            auto end = derive(256, 256, 256, 256, location, start, pathSize, angle::deg(i).rad(), velocity, 0.5 * 5.412658773);

            if (!graph->checkInGraph(end.x, end.y))
                FAIL();

            if (graph->getHeading(end.x, end.y) != angle::rad(end.z))
                FAIL();

            graph->clear();
        }
    }
    delete graph;
}