#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include "test_utils.h"

int convert_to_point(int x, int z)
{
    return 256 * z + x;
}

int convert_to_key(int x, int z)
{
    return 1000 * x + z;
}

TEST(TestGraph, TestCreateDelete)
{
    CudaGraph g(100, 100);
}

TEST(TestGraph, TestPhysicalParams)
{
    CudaGraph g(100, 100);
    g.setPhysicalParams(11.1, 11.2, angle::deg(21), 11.3, -1);
    double *p = g.getPhysicalParams();

    ASSERT_FLOAT_EQ(100/11.1, p[PHYSICAL_PARAMS_RATE_W]);
    ASSERT_FLOAT_EQ(11.1/100, p[PHYSICAL_PARAMS_INV_RATE_W]);
    ASSERT_FLOAT_EQ(100/11.2, p[PHYSICAL_PARAMS_RATE_H]);
    ASSERT_FLOAT_EQ(11.2/100, p[PHYSICAL_PARAMS_INV_RATE_H]);
    ASSERT_EQ(angle::deg(21), angle::rad(p[PHYSICAL_PARAMS_MAX_STEERING_RAD]));
    ASSERT_EQ(angle::deg(21), angle::deg(p[PHYSICAL_PARAMS_MAX_STEERING_DEG]));
    ASSERT_FLOAT_EQ((11.3/2), p[PHYSICAL_PARAMS_LR]);
}

TEST(TestGraph, TestSearchParams)
{
    CudaGraph g(100, 100);
    g.setSearchParams({4, 6}, {4, 5}, {6, 7});
    int * p = g.getSearchParams();

    ASSERT_EQ(2, p[FRAME_PARAM_MIN_DIST_X]);
    ASSERT_EQ(3, p[FRAME_PARAM_MIN_DIST_Z]);
    ASSERT_EQ(4, p[FRAME_PARAM_LOWER_BOUND_X]);
    ASSERT_EQ(5, p[FRAME_PARAM_LOWER_BOUND_Z]);
    ASSERT_EQ(6, p[FRAME_PARAM_UPPER_BOUND_X]);
    ASSERT_EQ(7, p[FRAME_PARAM_UPPER_BOUND_Z]);
}


TEST(TestGraph, TestSegmentationCosts)
{
    CudaGraph g(100, 100);
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {5}};
    g.setClassCosts(costs);
    float * p = g.getClassCosts();

    for (int i = 0; i < costs.size(); i++) {
        ASSERT_FLOAT_EQ(costs[i], p[i]);
    }
}
TEST(TestGraph, TestSegmentationCostsPtr)
{
    CudaGraph g(100, 100);
    float * costs = new float[6] {
        {0},
        {1},
        {2},
        {3},
        {4},
        {5}};
    g.setClassCosts(costs, 6);
    float * p = g.getClassCosts();

    for (int i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(costs[i], p[i]);
    }

    delete costs;
}

TEST(TestGraph, TestAddClear)
{
    CudaGraph g(100, 100);
    g.add(10,10, angle::deg(12), -1, -1, 101.2);
    ASSERT_EQ(1, g.countAll());
    ASSERT_TRUE(g.checkInGraph(10, 10));
    g.clear();
    ASSERT_FALSE(g.checkInGraph(10, 10));
    ASSERT_EQ(0, g.countAll());
}

TEST(TestGraph, TestAddClearTemporary)
{
    CudaGraph g(100, 100);
    g.addTemporary(10,10, angle::deg(12), -1, -1, 101.2);
    ASSERT_EQ(1, g.countAll());
    ASSERT_FALSE(g.checkInGraph(10, 10));
    g.clear();
    ASSERT_FALSE(g.checkInGraph(10, 10));
    ASSERT_EQ(0, g.countAll());
}

TEST(TestGraph, TestCoordinateStart)
{
    CudaGraph g(100, 200);
    float3 *p = g.getCoordinateStart();
    ASSERT_EQ(50, static_cast<int>(p->x));
    ASSERT_EQ(100, static_cast<int>(p->y));
    ASSERT_EQ(angle::deg(0), angle::rad(p->z));
    
    g.setCoordinateStart(30, 30);
    ASSERT_EQ(30, static_cast<int>(p->x));
    ASSERT_EQ(30, static_cast<int>(p->y));
    ASSERT_EQ(angle::deg(0), angle::rad(p->z));

    g.setCoordinateStart(40, 50, angle::deg(14.5));
    ASSERT_EQ(40, static_cast<int>(p->x));
    ASSERT_EQ(50, static_cast<int>(p->y));
    ASSERT_EQ(angle::deg(14.5), angle::rad(p->z));
}


TEST(TestGraph, TestBasicFeatures)
{
    CudaGraph g(100, 100);

    ASSERT_EQ(0, g.count());

    g.add(50, 50, angle::rad(0.12), -1, -1, 0);
    g.add(50, 45, angle::rad(0.12), 50, 50, 10);
    g.add(55, 40, angle::rad(0.12), 50, 45, 10);
    g.add(20, 35, angle::rad(0.12), 50, 45, 10);
    g.add(20, 15, angle::rad(0.12), 20, 35, 10);
    g.add(20, 5, angle::rad(0.12), 20, 15, 10);
    g.add(20, 0, angle::rad(0.12), 20, 5, 10);

    ASSERT_EQ(7, g.count());

    g.remove(20, 0);
    ASSERT_EQ(6, g.count());

    // remove non existant should ignore
    g.remove(0, 0);
    ASSERT_EQ(6, g.count());

    // remove of out-of-bound node should ignore
    g.remove(999, -999);
    ASSERT_EQ(6, g.count());

    // does not exist
    int2 p = g.getParent(20, 0);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // out-of-bound
    p = g.getParent(2000, -1000);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // exists
    p = g.getParent(20, 5);
    ASSERT_EQ(20, p.x);
    ASSERT_EQ(15, p.y);

    // setParent of non existant node should ignore
    g.setParent(99, 99, 10, 10);
    p = g.getParent(99, 99);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // setParent of non existant out-of-bound node should ignore
    g.setParent(999, 999, 10, 10);
    p = g.getParent(999, 999);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // exists
    ASSERT_FLOAT_EQ(g.getCost(55, 40), 10.0);

    // doesnt exist
    ASSERT_FLOAT_EQ(g.getCost(99, 99), 0);

    // out-of-bound
    ASSERT_FLOAT_EQ(g.getCost(999, 999), -1);

    g.setCost(55, 40, 20.0);
    ASSERT_FLOAT_EQ(g.getCost(55, 40), 20.0);

    // doesnt exist, but we do not verify to improve performance. The user is suposed to know what he's doing
    g.setCost(99, 99, 100.0);
    ASSERT_FLOAT_EQ(g.getCost(99, 99), 100);

    // out-of-bound
    g.setCost(999, 999, 100.0);
    ASSERT_FLOAT_EQ(g.getCost(999, 999), -1);
}

TEST(TestGraph, TestList)
{
    CudaGraph g(100, 100);

    ASSERT_EQ(0, g.count());
    std::unordered_map<int, double4> map;

    g.add(50, 50, angle::rad(0.12), -1, -1, 0);

    map[convert_to_key(50, 50)] = {0.12, -1, -1, 0};

    g.add(50, 45, angle::rad(0.11), 50, 50, 10);
    map[convert_to_key(50, 45)] = {0.11, 50, 50, 10};

    g.add(55, 40, angle::rad(0.10), 50, 45, 20);
    map[convert_to_key(55, 40)] = {0.10, 50, 45, 20};

    g.add(20, 35, angle::rad(0.09), 50, 45, 15);
    map[convert_to_key(20, 35)] = {0.09, 50, 45, 15};

    g.add(20, 15, angle::rad(0.08), 20, 35, 11);
    map[convert_to_key(20, 15)] = {0.08, 20, 35, 11};

    g.add(20, 5, angle::rad(0.07), 20, 15, 14);
    map[convert_to_key(20, 5)] = {0.07, 20, 15, 14};

    g.add(20, 0, angle::rad(-0.12), 20, 5, 99);
    map[convert_to_key(20, 0)] = {-0.12, 20, 5, 99};

    int count = g.count();

    ASSERT_EQ(7, count);
    std::vector<int2> res = g.list();

    for (int i = 0; i < count; i++)
    {
        int x = res[i].x;
        int z = res[i].y;
        int key = convert_to_key(x, z);
        ASSERT_TRUE(map.find(key) != map.end());
        angle heading = g.getHeading(res[i].x, res[i].y);

        double4 val = map[convert_to_key(x, z)];

        ASSERT_FLOAT_EQ(heading.rad(), val.x);

        int2 parent = g.getParent(x, z);

        ASSERT_EQ(parent.x, static_cast<int>(val.y));
        ASSERT_EQ(parent.y, static_cast<int>(val.z));

        double cost = g.getCost(x, z);
        ASSERT_FLOAT_EQ(cost, val.w);
    }
}

TEST(TestGraph, TestList_LoadTest)
{
    CudaGraph g(100, 100);

    ASSERT_EQ(0, g.count());
    std::unordered_map<int, double4> map;

    g.add(50, 50, angle::rad(0.12), -1, -1, 0);

    map[convert_to_key(50, 50)] = {0.12, -1, -1, 0};

    g.add(50, 45, angle::rad(0.11), 50, 50, 10);
    map[convert_to_key(50, 45)] = {0.11, 50, 50, 10};

    g.add(55, 40, angle::rad(0.10), 50, 45, 20);
    map[convert_to_key(55, 40)] = {0.10, 50, 45, 20};

    g.add(20, 35, angle::rad(0.09), 50, 45, 15);
    map[convert_to_key(20, 35)] = {0.09, 50, 45, 15};

    g.add(20, 15, angle::rad(0.08), 20, 35, 11);
    map[convert_to_key(20, 15)] = {0.08, 20, 35, 11};

    g.add(20, 5, angle::rad(0.07), 20, 15, 14);
    map[convert_to_key(20, 5)] = {0.07, 20, 15, 14};

    g.add(20, 0, angle::rad(-0.12), 20, 5, 99);
    map[convert_to_key(20, 0)] = {-0.12, 20, 5, 99};

    for (int c = 0; c < 1000; c++)
    {

        int count = g.count();
        if (count != 7)
            FAIL();

        for (int i = 0; i < count; i++)
        {
            std::vector<int2> res = g.list();
            int x = res[i].x;
            int z = res[i].y;
            int key = convert_to_key(x, z);
            if (map.find(key) == map.end())
                FAIL();

            angle heading = g.getHeading(res[i].x, res[i].y);

            double4 val = map[convert_to_key(x, z)];

            ASSERT_FLOAT_EQ(heading.rad(), val.x);

            int2 parent = g.getParent(x, z);

            ASSERT_EQ(parent.x, static_cast<int>(val.y));
            ASSERT_EQ(parent.y, static_cast<int>(val.z));

            double cost = g.getCost(x, z);
            ASSERT_FLOAT_EQ(cost, val.w);
        }
    }
}

TEST(TestGraph, TestList_LoadDump)
{
    CudaGraph g(100, 100);
    ASSERT_EQ(0, g.count());

    int parent_x = -1;
    int parent_z = -1;
    float cost = 0;

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
        {
            g.add(x, z, angle::rad(2), parent_x, parent_z, cost);
            cost++;
            parent_x = x;
            parent_z = z;
        }

    g.dumpGraph("/tmp/tmp34534.dat");

    g.clear();
    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
        {
            if (g.getType(x, z) != 0)
                FAIL();
        }

    g.readfromDump("/tmp/tmp34534.dat");

    cost = 0;
    parent_x = -1;
    parent_z = -1;    
    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
        {
            if (g.getType(x, z) != GRAPH_TYPE_NODE)
                FAIL();

            if (g.getHeading(x, z).rad() != 2)
                FAIL();

            if (g.getCost(x, z) != cost)
                FAIL();
            cost++;

            int2 p = g.getParent(x, z);
            if (p.x != parent_x && p.y != parent_z)
                FAIL();

            parent_x = x;
            parent_z = z;
        }
}
