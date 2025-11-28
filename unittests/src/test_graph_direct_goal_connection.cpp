#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>
#include <cuda_runtime.h>
#include <driveless/coord_conversion.h>
#include <driveless/world_pose.h>
#include <driveless/cuda_ptr.h>
#include <driveless/search_frame.h>
#include "test_utils.h"
#include <opencv2/opencv.hpp>

#define PHYS_SIZE 34.641016151377535

TEST(TestGraphGoalDirectConnection, FreeSpaceGoalConnection)
{
    CudaGraph *graph = buildTestGraph(40, 22);
    angle p = angle::rad(0);

    SearchFrame *f = buildTestSearchFrame();

    graph->addStart(128, 128, p); // A
    f->processDistanceToGoal(128, 0);
    f->processSafeDistanceZone({40, 22}, false);

    auto start = std::chrono::high_resolution_clock::now();
    graph->processDirectGoalConnection(f, 128, 0, angle::rad(0.0));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration / 1000 << " ms" << " (" << duration << ") us" << std::endl;
}

long computeFloatPos(int width, int x, int z)
{
    return 3 * (width * z + x);
}

TEST(TestGraphGoalDirectConnection, NarrowSpace)
{
    int min_x = 2;
    int min_z = 2;
    CudaGraph *graph = buildTestGraph(min_x, min_z);
    angle p = angle::rad(0);

    SearchFrame *f = new SearchFrame(256, 256, {-1, -1}, {-1, -1});
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {-1}};

    std::vector<std::tuple<int, int, int>> colors = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {255, 255, 255}};

    int row = 100;

    float *ptr = new float[256 * 256 * 3];
    for (int i = 0; i < 256 * 256 * 3; i++)
        ptr[i] = 0;

    for (int i = 0; i < 256; i++)
    {
        long pos = computeFloatPos(256, i, row);
        ptr[pos] = 5;
    }

    ptr[computeFloatPos(256, 127, row)] = 0;
    ptr[computeFloatPos(256, 128, row)] = 0;
    ptr[computeFloatPos(256, 129, row)] = 0;

    f->setClassCosts(costs);
    f->setClassColors(colors);
    f->copyFrom(ptr);
    delete[] ptr;

    graph->addStart(128, 128, p); // A
    f->processDistanceToGoal(128, 0);
    f->processSafeDistanceZone({min_x, min_z}, false);

    // uchar *dest = new uchar[3 * f->width() * f->height()];
    // f->exportToColorFrame(dest);
    // // Save dest as output.png using OpenCV
    // cv::Mat img(f->height(), f->width(), CV_8UC3, dest);
    // for (int z = 0; z < f->height(); z++)
    //     for (int x = 0; x < f->width(); x++)
    //     {
    //         long pos = 3 * (f->width() * z + x);
    //         if (!f->isTraversable(x, z))
    //         {
    //             dest[pos] = 128;
    //             dest[pos + 1] = 128;
    //             dest[pos + 2] = 128;
    //         }
    //     }

    // cv::imwrite("output.png", img);

    graph->processDirectGoalConnection(f, 128, 0, angle::rad(0.0));

    for (int i = 0; i < 256; i++)
    {
        if (i == 128)
        {
            ASSERT_TRUE(graph->isDirectlyConnectedToGoal(i, row));
        }
        else
        {
            ASSERT_FALSE(graph->isDirectlyConnectedToGoal(i, row));
        }

        // if (graph->isDirectlyConnectedToGoal(i, row))
        //     printf("%d,%d\n", i, row);
    }

    //     if (i == 112)
    //     {
    //         ASSERT_TRUE(graph->isDirectlyConnectedToGoal(i, row));
    //     }
    //     else
    //     {
    //         ASSERT_FALSE(graph->isDirectlyConnectedToGoal(i, row));
    //     }
    // }
}

TEST(TestGraphGoalDirectConnection, GoalIsUnfeasible)
{
    int min_x = 2;
    int min_z = 2;
    CudaGraph *graph = buildTestGraph(min_x, min_z);
    angle p = angle::rad(0);

    SearchFrame *f = new SearchFrame(256, 256, {-1, -1}, {-1, -1});
    std::vector<float> costs = {
        {-1},
        {1},
        {2},
        {3},
        {4},
        {5}};

    std::vector<std::tuple<int, int, int>> colors = {
        {255, 255, 255},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}};

    int row = 100;

    float *ptr = new float[256 * 256 * 3];
    for (int i = 0; i < 256 * 256 * 3; i++)
        ptr[i] = 1;

    ptr[computeFloatPos(256, 128, 0)] = 0;

    f->setClassCosts(costs);
    f->setClassColors(colors);
    f->copyFrom(ptr);
    delete[] ptr;

    graph->addStart(128, 128, p); // A
    f->processDistanceToGoal(128, 0);
    f->processSafeDistanceZone({min_x, min_z}, false);

    // uchar *dest = new uchar[3 * f->width() * f->height()];
    // f->exportToColorFrame(dest);
    // // Save dest as output.png using OpenCV
    // cv::Mat img(f->height(), f->width(), CV_8UC3, dest);
    // for (int z = 0; z < f->height(); z++)
    //     for (int x = 0; x < f->width(); x++)
    //     {
    //         long pos = 3 * (f->width() * z + x);
    //         if (!f->isTraversable(x, z))
    //         {
    //             dest[pos] = 128;
    //             dest[pos + 1] = 128;
    //             dest[pos + 2] = 128;
    //         }
    //     }

    // cv::imwrite("output.png", img);

    graph->processDirectGoalConnection(f, 128, 0, angle::rad(0.0));

    for (int i = 0; i < 256; i++)
    {
        ASSERT_FALSE(graph->isDirectlyConnectedToGoal(i, row));
    }

    //     if (i == 112)
    //     {
    //         ASSERT_TRUE(graph->isDirectlyConnectedToGoal(i, row));
    //     }
    //     else
    //     {
    //         ASSERT_FALSE(graph->isDirectlyConnectedToGoal(i, row));
    //     }
    // }
}