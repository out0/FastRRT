#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "test_utils.h"
#include "../../include/fastrrt.h"
#include "tst_class_def.h"
#include <driveless/search_frame.h>
// #include <driveless/cubic_interpolator.h>
#include <driveless/waypoint.h>
#include <string>
#include <chrono>

#define PHYS_SIZE 34.641016151377535

std::pair<cv::Mat, float *> readImg(const char *file)
{
    cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();
    float *ptr = new float[rows * cols * channels];

    for (int z = 0; z < rows; z++)
    {
        for (int x = 0; x < cols; x++)
        {
            int pos = 3 * (z * cols + x);
            cv::Vec3b pixel = img.at<cv::Vec3b>(z, x);
            ptr[pos] = pixel[0];
            ptr[pos + 1] = pixel[1];
            ptr[pos + 2] = pixel[2];

            // if (ptr[pos] > 0)
            //     printf ("x = %f\n", ptr[pos]);
        }
    }

    return std::pair<cv::Mat, float *>(img, ptr);
}

#define OG_REAL_WIDTH 34.641016151377535
#define OG_REAL_HEIGHT 34.641016151377535
#define MAX_STEERING_ANGLE 40
#define VEHICLE_LENGTH_M 5.412658774

void exportPathTo(float3 *f, int width, int height, std::vector<Waypoint> &path, const char *file)
{
    uchar *dest = new uchar[3 * width * height];

    SearchFrame frame(width, height, {-1, -1}, {-1, -1});
    frame.setClassColors(classColors);

    frame.setClassCosts(classCosts);

    frame.exportToColorFrame(dest);

    for (auto p : path)
    {
        int pos = 3 * (width * p.z() + p.x());
        dest[pos] = 0;
        dest[pos + 1] = 0;
        dest[pos + 2] = 255;
    }

    cv::Mat cimg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0));

    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
        {
            long pos = 3 * (h * width + w);
            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(h, w);
            pixel[0] = dest[pos];
            pixel[1] = dest[pos + 1];
            pixel[2] = dest[pos + 2];
        }

    cv::imwrite(file, cimg);
    delete[] dest;
}

void logGraph(FastRRT *rrt, SearchFrame *frame, const char *file, int i)
{
    int width = frame->width();
    int height = frame->height();
    uchar *dest = new uchar[3 * width * height];
    frame->exportToColorFrame(dest);

    cv::Mat cimg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0));

    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
        {
            long pos = 3 * (h * width + w);
            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(h, w);
            pixel[0] = dest[pos];
            pixel[1] = dest[pos + 1];
            pixel[2] = dest[pos + 2];
        }

    delete[] dest;

    std::vector<GraphNode> nodes = rrt->exportGraphNodes();
    // printf ("exporting %d nodes\n", nodes.size());

    for (auto n : nodes)
    {
        if (n.z < 0 || n.z >= height)
            continue;
        if (n.x < 0 || n.x >= width)
            continue;
        cv::Vec3b &pixel = cimg.at<cv::Vec3b>(n.z, n.x);

        switch (n.nodeType)
        {
        case GRAPH_TYPE_NODE:
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
            break;
        case GRAPH_TYPE_PROCESSING:
            pixel[0] = 0;
            pixel[1] = 255;
            pixel[2] = 0;
            break;
        case GRAPH_TYPE_TEMP:
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 255;
            break;
        default:
            break;
        }
    }

    cv::imwrite(file, cimg);
    std::string s = "log/graph_nodes_" + std::to_string(i) + ".txt";

    // Usage in logGraph
    exportGraphNodesToFile(nodes, s);
}

#define TIMEOUT -1

TEST(TestRRT, TestSearch)
{
    std::pair<cv::Mat, float *> res = readImg("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/libfastrrt/unittests/bev_1.png");
    cv::Mat img = res.first;
    float *ptr = res.second;

    // for (int i = 0; i < 10; i++)
    //     printf ("  %f, %f, %f", ptr[3*i], ptr[3*i+1], ptr[3*i+2]);
    // printf("\n");
    int height = img.rows;
    int width = img.cols;

    ASSERT_EQ(256, width);
    ASSERT_EQ(256, height);

    SearchFrame frame(width, height, {119, 148}, {137, 108});
    // SearchFrame frame(img.cols, img.rows, {22, 40}, {119, 148}, {137, 108});
    // frame.copyFrom(ptr);

    frame.setClassCosts(classCosts);
    frame.setClassColors(classColors);

    float maxPathSize = 40.0;
    float distToGoal = 20.0;

    frame.copyFrom(res.second);
    frame.processSafeDistanceZone({2, 2}, false);

    std::vector<float> p = classCosts;

    EgoParams egoParams = EgoParams::init(width, height)
                           .withEgoLowerBound({119, 148})
                           .withEgoUpperBound({137, 108})
                           .withSearchPhysicalSize(OG_REAL_WIDTH, OG_REAL_HEIGHT)
                           .withMaxSteeringAngle(angle::deg(MAX_STEERING_ANGLE))
                           .withVehicleLength(VEHICLE_LENGTH_M)
                           .withSegmentationClassCosts(classCosts)
                           .build();

    FastRRT rrt(egoParams);

    std::vector<Waypoint> path = rrt.getPlannedPath();

    ASSERT_EQ(path.size(), 0);

    Waypoint goal(107, 0, angle::rad(0));
    Waypoint start(128, 128, angle::rad(0));
    // rrt.setPlanData(frame, start, goal, 1, {11, 20});

    auto params = SearchParams::init(start, goal)
                      .withVelocity(1.0)
                      .withMinDistance({2, 2})
                      .withTimeout(TIMEOUT)
                      .withMaxPathSize(maxPathSize)
                      .withDistanceToGoalTolerance(distToGoal)
                      .withHeadingErrorTolerance(angle::rad(10))
                      .withFrame(&frame)
                      .build();

    auto chrono_start = std::chrono::high_resolution_clock::now();
    rrt.setPlanData(params);
    frame.processDistanceToGoal(goal.x(), goal.z());

    rrt.search_init();

    int i = 0;
    while (!rrt.goalReached() && rrt.loop(false))
    {
        // logGraph(&rrt, &frame, "output1.png", ++i);
    }

    ASSERT_TRUE(rrt.goalReached());

    path = rrt.getPlannedPath();
    // ASSERT_TRUE(path.size() > 5);

    auto last = path.back();
    int dx = last.x() - goal.x();
    int dz = last.z() - goal.z();

    ASSERT_LE(dx * dx + dz * dz, 400);

    // logGraph(&rrt, &frame, "/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/libfastrrt/tests/output1.png");
    rrt.path_optimize();
    // logGraph(&rrt, &frame, "/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/libfastrrt/tests/output1_optim.png");

    auto chrono_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_start).count();
    std::cout << "Execution time: " << duration / 1000 << " ms" << " (" << duration << ") us" << std::endl;

    // exportPathTo(frame.getFramePtr(), img.cols, img.rows, path, "output2.png");
    //  auto interpol_path = CubicInterpolator::cubicSplineInterpolation(path, 50);

    // rrt.optimize();
    // path = rrt.getPlannedPath();
    // //interpol_path = CubicInterpolator::cubicSplineInterpolation(path, 50);
    // exportPathTo(&frame, path, "output2.png");

    auto last2 = path.back();
    dx = last2.x() - goal.x();
    dz = last2.z() - goal.z();

    ASSERT_LE(dx * dx + dz * dz, 400);
    ASSERT_TRUE(rrt.goalReached());

    path = rrt.getPlannedPath();

    exportPathTo(frame.getCudaPtr(), img.cols, img.rows, path, "output2.png");

    delete[] ptr;
}
