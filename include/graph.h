#pragma once

#ifndef __GRAPH_DRIVELESS_H
#define __GRAPH_DRIVELESS_H

#include <driveless/search_frame.h>
#include <driveless/angle.h>
#include <driveless/cuda_ptr.h>
#include <driveless/cuda_frame.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>

class GraphNode
{
public:
    int x;
    int z;
    float heading_rad{};
    int nodeType;
    int parent_x{};
    int parent_z{};
    float connectToEndCost{};
    float cost{};

    GraphNode(int x, int z, int type) : x(x), z(z), nodeType(type) {}
    GraphNode() : x(0), z(0), heading_rad(0.0f), nodeType(0), parent_x(0), parent_z(0), connectToEndCost(0.0f), cost(0.0f) {}
};

#define GRAPH_TYPE_NULL 0
#define GRAPH_TYPE_NODE 1
#define GRAPH_TYPE_TEMP 2
#define GRAPH_TYPE_PROCESSING 3
#define GRAPH_TYPE_COLLISION 4
#define GRAPH_TYPE_CONNECT_TO_GOAL 5

#define THREADS_IN_BLOCK 256

typedef float3 pose;

class CudaGraph
{
private:
    std::shared_ptr<CudaFrame<int4>> _graph;
    std::shared_ptr<CudaFrame<float4>> _graphData;
    std::shared_ptr<CudaFrame<float4>> _graphCollision;
    std::shared_ptr<CudaFrame<float3>> _graphGoalDirectConnection;
    bool __checkLimits(int x, int z);

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

    std::tuple<int, float> __findFirstDirectConnectionToPos(float3 *og, std::vector<float4> res, int pos, bool isSafeZoneChecked);
    std::vector<float4> __getPlannedPath(float3 *og, int2 goal, angle goalHeading, float distanceToGoalTolerance);
    

public:
    CudaGraph(int width, int height);
    ~CudaGraph();

    void computeGraphRegionDensity();

    void computeRepulsiveFieldAPF(float3 *og, float Kr, int radius);
    void computeAttractiveFieldAPF(float3 *og, float Ka, std::pair<int, int> goal);

    void setPhysicalParams(float perceptionWidthSize_m, float perceptionHeightSize_m, angle maxSteeringAngle, float vehicleLength, float max_curvature);
    double *getPhysicalParams()
    {
        return _physicalParams->get();
    }

    void setSearchParams(std::pair<int, int> minDistance, std::pair<int, int> lowerBound, std::pair<int, int> upperBound);
    int *getSearchParams()
    {
        return _searchSpaceParams->get();
    }

    void setClassCosts(float *costs, int count);
    void setClassCosts(std::vector<float> costs);
    float *getClassCosts()
    {
        return _classCosts->get();
    }
    unsigned int getClassCount()
    {
        return _classCosts->count();
    }

    void add(int x, int z, angle heading, int parent_x, int parent_z, float cost);
    void addTemporary(int x, int z, angle heading, int parent_x, int parent_z, float cost);

    void setCoordinateStart(int x, int z, angle heading);
    void setCoordinateStart(int x, int z);
    void addStart(int x, int z, angle heading);
    float3 *getCoordinateStart();

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
    std::shared_ptr<CudaFrame<int4>> getFramePtr()
    {
        return _graph;
    }
    std::shared_ptr<CudaFrame<float4>> getFrameDataPtr()
    {
        return _graphData;
    }

    std::shared_ptr<CudaFrame<float3>> getDirectConnectionDataPtr()
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
    int2 derivateNode(float3 *og, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z);

    /// @brief Derivates all nodes in graph with a random steering angle and pathSize, for the specified maxSteeringAngle, maxPathSize, and velocity_m_s.
    /// @param maxSteeringAngle
    /// @param maxPathSize
    /// @param velocity_m_s
    void expandTree(float3 *og, angle goalHeading, float maxPathSize, float velocity_m_s, bool frontierExpansion, int2 start_node, int2 goal, angle goal_heading);

    void smartExpansion(float3 *og, angle goalHeading, float maxPathSize, float velocity_m_s, bool expandFrontier, bool forceExpand, int2 goal, angle goal_heading);

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
    int2 findBestNode(float3 *og, angle heading, float radius, int x, int z, float maxHeadingError);

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

    /// @brief Finds the cells that can direcly connect to the goal using hermite. Accounts for collision detection and max curvature.
    /// @param frame
    /// @param goal_x
    /// @param goal_z
    /// @param goal_heading
    void processDirectGoalConnection(SearchFrame *frame, int goal_x, int goal_z, angle goal_heading, float max_curvature = -1);

    /// @brief Returns true if the cell x,z can direcly connect to the goal (via processDirectGoalConnection)
    /// @param x 
    /// @param z 
    /// @return 
    bool isDirectlyConnectedToGoal(int x, int z);

    /// @brief Returns the cost of cell x,z direcly connected to the goal (via processDirectGoalConnection)
    /// @param x 
    /// @param z 
    /// @return 
    float directConnectionToGoalCost(int x, int z);

    /// @brief Returns the heading of cell x,z direcly connected to the goal (via processDirectGoalConnection)
    /// @param x 
    /// @param z 
    /// @return 
    angle directConnectionToGoalHeading(int x, int z);


    bool findBestGoalDirectConnection(float3 *og, float radius, bool isSafeZoneChecked);

    float4 bestGraphDirectConnectionParent();

    float4 bestGraphDirectConnectionChild();

    sptr<float4> convertPlannedPath(std::vector<Waypoint> path);

    bool optimizePathLoop(float3 *og, sptr<float4> path, int path_size, float distanceToGoalTolerance, bool isSafeZoneChecked);
};

#endif