#pragma once

#ifndef __CUDA_GRAPH_DRIVELESS_H
#define __CUDA_GRAPH_DRIVELESS_H

#include <driveless/search_frame.h>
#include <driveless/angle.h>
#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include <driveless/frame.h>
#include <vector>
#include <atomic>
#include <memory>
#include "graph_node.h"

#define BLOCK_SIZE 128

#ifdef DRIVELESS_CUDA_ENABLED
#include <cuda_runtime.h>
#include <curand_kernel.h>
#else
struct RandState
{
    uint32_t s[4];
};
#endif

class CudaGraph
{
private:
    std::shared_ptr<Frame<int4>> _graph;
    std::shared_ptr<Frame<float4>> _graphData;
    std::shared_ptr<Frame<float4>> _graphCollision;
    std::shared_ptr<Frame<float3>> _graphGoalDirectConnection;
    bool __checkLimits(int x, int z);

#ifdef DRIVELESS_CUDA_ENABLED
    cptr<float3> _ogCoordinateStart;
    cptr<unsigned int> _parallelCount;
    cptr<bool> _newNodesAdded;
    cptr<bool> _nodeCollision;
    cptr<bool> _goalReached;
    cptr<double> _physicalParams;
    cptr<int> _searchSpaceParams;
    cptr<unsigned int> _region_node_count;
    cptr<float> _classCosts;
    cptr<curandState> _randState;
    // find best node for direct connection
    cptr<float4> _bestNodeDirectConnection;
    cptr<long long> _bestNodeDirectConnectionCost;
#else
    float3 _ogCoordinateStart;
    unsigned int _parallelCount;
    std::atomic<bool> _newNodesAdded{false};
    std::atomic<bool> _nodeCollision{false};
    std::atomic<bool> _goalReached{false};
    std::unique_ptr<double[]> _physicalParams;
    std::unique_ptr<int[]> _searchSpaceParams;
    std::unique_ptr<unsigned int[]> _region_node_count;
    std::unique_ptr<float[]> _classCosts;
    int _classCostsCount{0};
    std::unique_ptr<RandState[]> _randState;
    // find best node for direct connection
    std::unique_ptr<float4[]> _bestNodeDirectConnection;
    long long _bestNodeDirectConnectionCost;
#endif
    void __initializeRandomGenerator();
    std::pair<int2 *, int> __listNodes(int type);
    std::pair<int3 *, int> __listAllNodes();

    void __initializeRegionDensity();
    void __dealocRegionDensity();

    float _node_mean;
    int _directOptimPos;

    void __printInconsistentChain(int3 n, int maxLoop);

    unsigned int __countInRange(int xp, int zp, float radius_sqr);
    std::pair<int2 *, int> __listNodesInRange(int type, int x, int z, float radius);

public:
    CudaGraph(int width, int height);
    ~CudaGraph();

    void computeGraphRegionDensity();

    void computeRepulsiveFieldAPF(float3 *og, float Kr, int radius);
    void computeAttractiveFieldAPF(float3 *og, float Ka, std::pair<int, int> goal);

    void setPhysicalParams(float perceptionWidthSize_m, float perceptionHeightSize_m, angle maxSteeringAngle, float vehicleLength, float max_curvature);
    double *getPhysicalParams()
    {
#ifdef DRIVELESS_CUDA_ENABLED
        return _physicalParams->get();
#else
        return _physicalParams.get();
#endif
    }

    void setSearchParams(std::pair<int, int> minDistance, std::pair<int, int> lowerBound, std::pair<int, int> upperBound);
    int *getSearchParams()
    {
#ifdef DRIVELESS_CUDA_ENABLED
        return _searchSpaceParams->get();
#else
        return _searchSpaceParams.get();
#endif
    }

    void setPreProcessCollisionEnable(bool vectorCheck);
    void setPreProcessDistanceEnable();

    void setClassCosts(float *costs, int count);
    void setClassCosts(std::vector<float> costs);
    float *getClassCosts()
    {
#ifdef DRIVELESS_CUDA_ENABLED
        return _classCosts->get();
#else
        return _classCosts.get();
#endif
    }
    unsigned int getClassCount()
    {
#ifdef DRIVELESS_CUDA_ENABLED
        return _classCosts->count();
#else
        return _classCostsCount;
#endif
    }

    void add(int x, int z, angle heading, int parent_x, int parent_z, float cost);
    void addTemporary(int x, int z, angle heading, int parent_x, int parent_z, float cost);

    void setCoordinateStart(int x, int z, angle heading);
    void setCoordinateStart(int x, int z);
    void addStart(int x, int z, angle heading);
    float3 getCoordinateStart();

    void remove(int x, int z);
    void clear();
    std::vector<int2> list();
    std::vector<int3> listAll();
    std::vector<int2> listInRange(int x, int z, float radius);
    unsigned int count(int type = GRAPH_TYPE_NODE);
    unsigned int countAll();

    inline int height()
    {
        return _graph->height();
    }
    inline int width()
    {
        return _graph->width();
    }
    std::shared_ptr<Frame<int4>> getFramePtr()
    {
        return _graph;
    }
    std::shared_ptr<Frame<float4>> getFrameDataPtr()
    {
        return _graphData;
    }

    std::shared_ptr<Frame<float3>> getDirectConnectionDataPtr()
    {
        return _graphGoalDirectConnection;
    }

    // int2 getCenter() {
    //     return _gridCenter;
    // }

    bool checkInGraph(int x, int z);
    void setParent(int x, int z, int parent_x, int parent_z);
    int2 getParent(int x, int z);
    angle getHeading(int x, int z);
    void setHeading(int x, int z, angle heading);
    float getCost(int x, int z);
    void setCost(int x, int z, float cost);

    void setType(int x, int z, int type);

    int getType(int x, int z);

    /// @brief Derivates a node on position {x, z} for the specified steeringAngle, pathSize and velocity_m_s. The node must exist in the graph.
    /// @param x
    /// @param z
    /// @param heading
    /// @return final node of the path
    float4 derivateNode(float3 *og, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z);

    /// @brief Derivates all nodes in graph with a random steering angle and pathSize, for the specified maxSteeringAngle, maxPathSize, and velocity_m_s.
    /// @param maxSteeringAngle
    /// @param maxPathSize
    /// @param velocity_m_s
    void expandTree(float3 *og, float maxPathSize, float velocity_m_s,
                    bool controlExpansion, bool forceExpansion, int2 goal, angle goal_heading,
                    float dist_to_goal_tolerance, angle heading_error_tolerance);

    void smartExpansion(float3 *og, float maxPathSize, float velocity_m_s,
                        bool controlExpansion, bool forceExpansion, int2 goal,
                        angle goal_heading, float dist_to_goal_tolerance, angle heading_error_tolerance);

    /// @brief Accepts a derivated node and connects it to the graph.
    /// @param start
    /// @param lastNode
    /// @return true for accepted nodes, false otherwise
    void acceptDerivedNode(int2 start, int2 lastNode);

    /// @brief Accepts all derivated nodes and connects them to the graph.
    /// @return
    void acceptDerivedNodes(int2 goal, float goal_heading);

    /// @brief Finds the best node in graph (with the lowest cost) that is feasible with the given heading, in a given search radius
    /// @param searchFrame
    /// @param radius
    /// @param x
    /// @param z
    /// @param heading
    /// @return
    long long findBestNodeCost(float3 *og, angle heading, float radius, int x, int z, float maxHeadingError);

    /// @brief Finds the best node in graph (with the lowest cost) that is feasible with the given heading, in a given search radius
    /// @param searchFrame
    /// @param radius
    /// @param x
    /// @param z
    /// @param heading
    /// @return
    int2 findBestNode(float3 *og, angle heading, float radius, int x, int z, float maxHeadingError, long long cost);

    /// @brief Checks if there is a feasible connection between start and end, at the given velocity and max steering angle
    /// @param searchFrame
    /// @param start
    /// @param end
    /// @param velocity_m_s
    /// @param maxSteeringAngle
    /// @return
    bool checkFeasibleConnection(float3 *og, int2 start, int2 end, int velocity_m_s);

    /// @brief Returns true if any node in the graph is at a distance equals or lower than distanceToGoalTolerance and is feasible on the given heading.
    /// @param searchFrame
    /// @param goal
    /// @param heading
    /// @param distanceToGoalTolerance
    /// @return
    bool checkGoalReached(float3 *og, int2 goal, angle heading, float distanceToGoalTolerance, float maxHeadingError);

    void dumpGraph(const char *filename);

    void readfromDump(const char *filename);

    bool checkNewNodesAddedOnTreeExpansion();

    void solveCollisions();

    bool canConnectToGoal(SearchFrame *frame, int x, int z, int goal_x, int goal_z, int goal_heading);

    /// @brief Returns true if the GRAPH is DAG consistent. This is usually be used for testing and debugging, as bugfree operation should always be DAG consistent
    /// @return
    bool checkGraphIsConsistent(bool print_inconsistency = true);

    /// @brief Returns the number of childs of the node x, z
    /// @param x
    /// @param z
    /// @return
    int getChildCount(int x, int z);

    void setCollision(int x, int z, int new_parent_x, int new_parent_z, angle new_heading, float new_cost);

    float getDirectCost(int x, int z);

    void dumpNodesToFile(const char *filename);

#ifdef DRIVELESS_CUDA_ENABLED
    sptr<float4> convertPlannedPath(std::vector<Waypoint> path);

    bool optimizePathLoop(float3 *frame, sptr<float4> path, int path_size, float distanceToGoalTolerance);
#else
    std::shared_ptr<float4[]> convertPlannedPath(std::vector<Waypoint> path);

    bool optimizePathLoop(float3 *frame, std::shared_ptr<float4[]> path, int path_size, float distanceToGoalTolerance);
#endif
};

#endif