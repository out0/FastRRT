#include <cmath>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../include/graph.h"

void exportGraph2(CudaGraph *graph, const char *filename)
{
    cv::Mat cimg = cv::Mat(graph->height(), graph->width(), CV_8UC3, cv::Scalar(0));

    int3 *ptr = graph->getFramePtr()->getCudaPtr();

    for (int h = 0; h < graph->height(); h++)
        for (int w = 0; w < graph->width(); w++)
        {
            if (w == 128 && h == 128)  {
                int zz = 1;
            }
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

    cv::imwrite(filename, cimg);
}