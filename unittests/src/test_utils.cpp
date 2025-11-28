#include <cmath>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include "test_utils.h"

bool _ASSERT_DEQ(double a, double b, int tolerance)
{
    double p = pow(10, -tolerance);

    if (abs(a - b) > p)
    {
        printf("ASSERT_DEQ failed: %f != %f, tolerance: %f\n", a, b, p);
        return false;
    }

    return true;
}

std::vector<int2> get_planned_path(CudaGraph *graph, float3 *ptr, angle goal_heading, int goal_x, int goal_z, float distToGoalTolerance)
{
    // res.push_back(*_goal);
    int2 n = graph->findBestNode(ptr, goal_heading, distToGoalTolerance, goal_x, goal_z, TO_RAD * 10);
    std::vector<int2> res;

    while (n.x != -1 && n.y != -1)
    {
        res.push_back({n.x, n.y});
        n = graph->getParent(n.x, n.y);
    }

    std::reverse(res.begin(), res.end());
    return res;
}

void exportGraph(CudaGraph *graph, const char *filename, std::vector<int2> *path)
{
    cv::Mat cimg = cv::Mat(graph->height(), graph->width(), CV_8UC3, cv::Scalar(0));

    int4 *ptr = graph->getFramePtr()->getCudaPtr();

    for (int h = 0; h < graph->height(); h++)
        for (int w = 0; w < graph->width(); w++)
        {
            long pos = h * graph->width() + w;
            if (ptr[pos].z == 0)
                continue;

            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(h, w);

            switch (ptr[pos].z)
            {
            case GRAPH_TYPE_NODE:
                pixel[0] = 255;
                pixel[1] = 255;
                pixel[2] = 255;
                break;
            case GRAPH_TYPE_TEMP:
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = 0;
            case GRAPH_TYPE_PROCESSING:
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            default:
                break;
            }
        }

    if (path != nullptr)
    {
        for (auto p : *path)
        {
            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(p.y, p.x);
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 255;
        }
    }

    cv::imwrite(filename, cimg);
}

CudaPtr<float3> createEmptySearchFrame(int width, int height)
{
    CudaPtr<float3> ptr(width * height);
    long size = height * width;
    for (int i = 0; i < size; i++)
    {
        ptr.get()[i].x = 0;
        ptr.get()[i].y = 0;
        ptr.get()[i].z = 0;
    }
    return ptr;
}

SearchFrame *createEmptySearchFramePtr(int width, int height)
{
    auto ptr = new SearchFrame(width, height, {-1, -1}, {-1, -1});
    long size = height * width;
    float *p = new float[size * 3];

    for (int i = 0; i < size; i++)
    {
        int pos = 3 * i;
        p[pos] = 0;
        p[pos + 1] = 0;
        p[pos + 2] = 0;
    }
    ptr->copyFrom(p);
    delete[] p;
    return ptr;
}

void assertInt2Equal(int2 a, int2 b)
{
    if (a.x != b.x || a.y != b.y)
    {
        printf("(%d, %d) != (%d, %d)\n", a.x, a.y, b.x, b.y);
        FAIL();
    }
}

CudaGraph *buildTestGraph(int min_dist_x, int min_dist_z)
{
    CudaGraph *g = new CudaGraph(256, 256);
    angle maxSteering = angle::deg(40);
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {-1}};

    g->setPhysicalParams(256, 256, maxSteering, 5.412658773, -1);
    g->setClassCosts(costs);
    g->setSearchParams({min_dist_x, min_dist_z}, {-1, -1}, {-1, -1});
    return g;
}
SearchFrame *buildTestSearchFrame()
{
    SearchFrame *f = new SearchFrame(256, 256, {-1, -1}, {-1, -1});
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {-1}};

    float *ptr = new float[256 * 256 * 3];
    for (int i = 0; i < 256*256*3; i++)
        ptr[i] = 0;

    f->setClassCosts(costs);
    f->copyFrom(ptr);
    delete []ptr;
    return f;
}

// Export graph nodes to a file
void exportGraphNodesToFile(const std::vector<GraphNode> &nodes, const std::string &filename)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open())
        return;

    for (const auto &n : nodes)
    {
        ofs << n.x << " "
            << n.z << " "
            << n.heading_rad << " "
            << n.nodeType << " "
            << n.parent_x << " "
            << n.parent_z << " "
            << n.connectToEndCost << " "
            << n.cost << "\n";
    }
    ofs.close();
}

// Import graph nodes from a file
std::vector<GraphNode> importGraphNodesFromFile(const std::string &filename)
{
    std::vector<GraphNode> nodes;
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        return nodes;

    GraphNode n;
    while (ifs >> n.x >> n.z >> n.heading_rad >> n.nodeType >> n.parent_x >> n.parent_z >> n.connectToEndCost >> n.cost)
    {
        nodes.push_back(n);
    }
    ifs.close();
    return nodes;
}

#include <sstream>

std::vector<GraphNode> importGraphNodesFromString(const std::string &data)
{
    std::vector<GraphNode> nodes;
    std::istringstream iss(data);

    GraphNode n;
    while (iss >> n.x >> n.z >> n.heading_rad >> n.nodeType >> n.parent_x >> n.parent_z >> n.connectToEndCost >> n.cost)
    {
        nodes.push_back(n);
    }
    return nodes;
}