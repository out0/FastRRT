#pragma once

#ifndef __FASTRRT_DRIVELESS_H
#define __FASTRRT_DRIVELESS_H

#include <cmath>
#include <chrono>
#include <driveless/angle.h>
#include <driveless/waypoint.h>
#include <driveless/cuda_frame.h>
#include <driveless/search_params.h>
#include <vector>
#include "graph.h"

// typedef float3* cudaPtr;

class FastRRT
{
private:
    CudaGraph _graph;
    std::chrono::time_point<std::chrono::high_resolution_clock> _exec_start;
    long _timeout_ms;
    float _maxPathSize;
    float _distToGoalTolerance;
    Waypoint _start;
    Waypoint _goal;
    float3 *_ptr;
    float _planningVelocity_m_s;
    int2 _bestNode;
    int _last_expanded_node_count;
    bool _hasPlanData;
    angle _headingErrorTolerance;
    EgoParams _egoParams;

    void __set_exec_started();
    long __get_exec_time_ms();
    bool __check_timeout();
    void __shrink_search_graph();

public:
    FastRRT(EgoParams &egoParams);

    void setPlanData(SearchParams &params);

    /// @brief
    /// @param copyIntrinsicCostsFromFrame copys the values in frame's channel G as intrinsic values to support using cost maps.
    void search_init(bool copyIntrinsicCostsFromFrame = false);
    bool loop(bool smartExpansion = false);
    bool path_optimize();
    bool goalReached();

    /// @brief Exports the current state of the graph as a vector
    /// @return vector, where each node = [x, z, node_type]
    std::vector<GraphNode> exportGraphNodes();

    std::vector<Waypoint> getPlannedPath();
    std::vector<Waypoint> interpolatePlannedPath();
    std::vector<Waypoint> interpolatePlannedPath(std::vector<Waypoint> path);
    std::vector<Waypoint> idealGeometryCurveNoObstacles(Waypoint goal);
    void computeGraphRegionDensity();

    /// @brief Saves the planner current Graph state, to be used when debugging the algorithm execution
    /// @param filename
    void saveCurrentGraphState(std::string filename);

    /// @brief Loads planner current Graph state from file, to be used when debugging the algorithm execution
    /// @param filename
    void loadGraphState(std::string filename);
};

#endif