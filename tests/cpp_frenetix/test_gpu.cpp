#include <driveless/fastrrt.h>
#include <driveless/search_params.h>
#include <driveless/search_frame.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <utility>

const int PERCEPTION_WIDTH_M = 1;
const int PERCEPTION_HEIGHT_M = 1;
const float VEHICLE_LENGTH_M = 5.412658774;
const int TIMEOUT = -1;

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
    // exportGraphNodesToFile(nodes, s);
}

int main()
{
    std::pair<cv::Mat, float *> res = readImg("converted_bev_23.png");
    cv::Mat img = res.first;
    float *ptr = res.second;

    EgoParams egoParams = EgoParams::init(img.cols, img.rows)
                              .withEgoLowerBound({119, 148})
                              .withEgoUpperBound({137, 108})
                              .withSearchPhysicalSize(PERCEPTION_WIDTH_M, PERCEPTION_HEIGHT_M)
                              .withMaxSteeringAngle(angle::deg(40))
                              .withVehicleLength(VEHICLE_LENGTH_M)
                              .withSegmentationClassCosts({-1, 0, 0, 0, 0})
                              .withSegmentationClassColors({{0, 0, 0},
                                                            {255, 255, 255},
                                                            {255, 255, 255},
                                                            {255, 255, 255},
                                                            {255, 255, 255}})
                              .build();

    auto start = Waypoint(416, 686, angle::deg(90 + -0.039754376));
    auto goal = Waypoint(296, 15, angle::deg(-14));

    SearchFrame *frame = egoParams.newSearchFrame();

    auto params = egoParams.newSearchParams(start, goal)
                      .withVelocity(1.0)
                      .withMinDistance({2, 2})
                      .withTimeout(TIMEOUT)
                      .withMaxPathSize(40.0)
                      .withDistanceToGoalTolerance(20.0)
                      .withHeadingErrorTolerance(angle::rad(10))
                      .withFrame(frame)
                      .build();
    frame->copyFrom(ptr);

    FastRRT rrt(egoParams);
    rrt.setPlanData(params);

    frame->processSafeDistanceZone({2, 2}, false);
    frame->processDistanceToGoal(goal.x(), goal.z());

    rrt.search_init(true);

    int loop_count = 0;
    while (!rrt.goalReached() && rrt.loop(true))
    {
        loop_count++;
    }
    auto path = rrt.getPlannedPath();

    printf("found a path with size %ld\n", path.size());

    auto interpol_path = rrt.interpolatePlannedPath();

    uchar *dest = new uchar[frame->width() * frame->height() * 3];
    frame->exportToColorFrame(dest);
    int width = frame->width();
    for (auto p : interpol_path)
    {
        long pos = 3 * (p.z() * width + p.x());
        dest[pos] = 255;
        dest[pos + 1] = 0;
        dest[pos + 2] = 0;
    }
    {
        int w = frame->width();
        int h = frame->height();
        cv::Mat out(h, w, CV_8UC3, dest);
        cv::imwrite("output1.png", out);
        delete[] dest;
    }

    return 0;

    // auto ego = EgoParams::init(img.cols, img.rows)
    //                .withSegmentationClassCosts(std::vector<float>{-1.f, 0.f, 0.f, 0.f, 0.f})
    //                .withSegmentationClassColors(std::vector<std::tuple<int, int, int>>{
    //                    {0, 0, 0},
    //                    {128, 128, 128},
    //                    {128, 128, 128},
    //                    {128, 128, 128},
    //                    {128, 128, 128}})
    //             //    .withSegmentationClassColors(std::vector<std::tuple<int, int, int>>{
    //             //        {0, 0, 0},
    //             //        {128, 128, 128},
    //             //        {0, 0, 255},
    //             //        {255, 255, 255},
    //             //        {255, 0, 0}})
    //                .withEgoLowerBound({-1, -1})
    //                .withEgoUpperBound({-1, -1})
    //                .withMaxSteeringAngle(angle::deg(40))
    //                .withMaxCurvature(0.34)
    //                .withSearchPhysicalSize(PERCEPTION_WIDTH_M, PERCEPTION_HEIGHT_M)
    //                .withVehicleLength(VEHICLE_LENGTH_M)
    //                .build();

    // auto frame = ego.newSearchFrame();
    // frame->copyFrom(ptr);
    // frame->processDistanceToGoal(296, 15);
    // frame->processSafeDistanceZone({2, 2}, false);

    // uchar *dest = new uchar[frame->width() * frame->height() * 3];
    // frame->exportToColorFrame(dest);
    // {
    //     int w = frame->width();
    //     int h = frame->height();
    //     cv::Mat out(h, w, CV_8UC3, dest);
    //     cv::imwrite("output1.png", out);
    //     delete[] dest;
    // }
    // return 0;

    // auto search_params = ego.newSearchParams(
    //                             Waypoint(416, 686, angle::deg(90 + -0.039754376)),
    //                             Waypoint(296, 15, angle::deg(-14)))
    //                          .withDistanceToGoalTolerance(0.0)
    //                          .withFrame(frame)
    //                          .withMaxPathSize(40.0)
    //                          .withMinDistance(std::make_pair(2, 2))
    //                          .withVelocity(1.0)
    //                          .withTimeout(-1)
    //                          .build();

    // FastRRT rrt(ego);
    // rrt.setPlanData(search_params);

    // rrt.search_init(true);
    // while (rrt.loop(true) && !rrt.goalReached()) {
    //     //logGraph(&rrt, &frame, "output1.png", ++i);
    // }

    // auto path = rrt.getPlannedPath();

    // printf ("found a path with %d points", path.size());

    // return 0;
}