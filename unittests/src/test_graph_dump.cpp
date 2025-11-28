#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include "test_utils.h"

TEST(TestDumpGraph, TestDumpRestore)
{
    CudaGraph g(100, 100);
    CudaGraph g2(100, 100);
    g.add(50, 50, angle::deg(0), -1, -1, 0);
    g.add(40, 40, angle::deg(-11), 50, 50, 10);
    g.addTemporary(30, 30, angle::deg(4), 40, 40, -20);
    ASSERT_EQ(3, g.countAll());

    g.dumpGraph("test.dat");
    g.clear();
    ASSERT_EQ(0, g.countAll());

    g2.readfromDump("test.dat");
    ASSERT_EQ(3, g2.countAll());

    auto nodes = g2.listAll();

    int found[3];

    found[0] = 0;
    found[1] = 0;
    found[2] = 0;

    for (int3 n : nodes) {
        if (n.x == 50) {
            found[0]++;
            ASSERT_EQ(n.z, GRAPH_TYPE_NODE);
            int2 p = g2.getParent(n.x, n.y);
            ASSERT_EQ(-1, p.x);
            ASSERT_EQ(-1, p.y);
            ASSERT_EQ(0, g2.getCost(n.x, n.y));
        }
        if (n.x == 40) {
            found[1]++;
            ASSERT_EQ(n.z, GRAPH_TYPE_NODE);
            int2 p = g2.getParent(n.x, n.y);
            ASSERT_EQ(50, p.x);
            ASSERT_EQ(50, p.y);
            ASSERT_EQ(10, g2.getCost(n.x, n.y));
        }
        if (n.x == 30) {
            found[2]++;
            ASSERT_EQ(n.z, GRAPH_TYPE_TEMP);
            int2 p = g2.getParent(n.x, n.y);
            ASSERT_EQ(40, p.x);
            ASSERT_EQ(40, p.y);
            ASSERT_EQ(-20, g2.getCost(n.x, n.y));
        }
        ASSERT_EQ(n.x, n.y);
    }

    ASSERT_EQ(1, found[0]);
    ASSERT_EQ(1, found[1]);
    ASSERT_EQ(1, found[2]);
}