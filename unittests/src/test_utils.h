#ifndef __TEST_UTILS_DRIVELESS_H
#define __TEST_UTILS_DRIVELESS_H

#include "../../include/graph.h"
#include <driveless/cuda_ptr.h>
#include <driveless/search_frame.h>
#include <gtest/gtest.h>

extern bool _ASSERT_DEQ(double a, double b, int tolerance = 4);
#define ASSERT_DEQ(a, b) ASSERT_TRUE(_ASSERT_DEQ(a, b))

std::vector<int2> get_planned_path(CudaGraph *graph, float3 *ptr, angle goal_heading, int goal_x, int goal_z, float distToGoalTolerance);

void exportGraph(CudaGraph *graph, const char *filename, std::vector<int2> *path = nullptr);

CudaPtr<float3> createEmptySearchFrame(int width, int height);

SearchFrame *createEmptySearchFramePtr(int width, int height);

void assertInt2Equal(int2 a, int2 b);

CudaGraph *buildTestGraph(int min_dist_x = 0, int min_dist_z = 0);
SearchFrame *buildTestSearchFrame();

// Export graph nodes to a file
void exportGraphNodesToFile(const std::vector<GraphNode> &nodes, const std::string &filename);

// Import graph nodes from a file
std::vector<GraphNode> importGraphNodesFromFile(const std::string &filename);


std::vector<GraphNode> importGraphNodesFromString(const std::string& data);
#endif