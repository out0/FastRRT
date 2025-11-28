#include "../include/fastrrt.h"
#include <bits/algorithmfwd.h>
/*
FastRRT::FastRRT(
    int width,
    int height,
    float perceptionWidthSize_m,
    float perceptionHeightSize_m,
    angle maxSteeringAngle,
    float vehicleLength,
    angle headingErrorTolerance,
    float max_curvature) : _graph(CudaGraph(width, height)),
                           _start(Waypoint(0, 0, angle::rad(0))),
                           _goal(Waypoint(0, 0, angle::rad(0))),
                           _hasPlanData(false),
                           _headingErrorTolerance(headingErrorTolerance)
{
    // printf ("Parameters: \n");
    // printf ("width: %d, height: %d\n", width, height);
    // printf ("perception width: %f, height: %f\n", perceptionWidthSize_m, perceptionHeightSize_m);
    // printf ("max steering: deg %f, rad: %f\n", maxSteeringAngle.deg(), maxSteeringAngle.rad());
    // printf ("vehicleLength = %f\n", vehicleLength);
    // printf ("timeout_ms = %d\n", timeout_ms);
    // printf ("minDistance = %d, %d\n", minDistance.first, minDistance.second);
    // printf ("lowerBound = %d, %d\n", lowerBound.first, lowerBound.second);
    // printf ("upperBound = %d, %d\n", upperBound.first, upperBound.second);
    // printf ("maxPathSize = %f\n", maxPathSize);
    // printf ("distToGoalTolerance = %f\n", distToGoalTolerance);

    _graph.setPhysicalParams(perceptionWidthSize_m, perceptionHeightSize_m, maxSteeringAngle, vehicleLength, max_curvature);
    _ptr = nullptr;
}
*/

FastRRT::FastRRT(EgoParams &egoParams) : _graph(CudaGraph(egoParams.width(), egoParams.height())),
                           _start(Waypoint(0, 0, angle::rad(0))),
                           _goal(Waypoint(0, 0, angle::rad(0))),
                           _hasPlanData(false),
                           _headingErrorTolerance(angle::deg(10)),
                           _egoParams(egoParams)
{
    auto [perceptionWidthSize_m, perceptionHeightSize_m] = egoParams.searchFramePhysicalDimensions();
    _graph.setPhysicalParams(perceptionWidthSize_m, perceptionHeightSize_m, egoParams.maxSteeringAngle(), egoParams.vehicleLength_m(), egoParams.maxCurvature());
    _ptr = nullptr;
}

void FastRRT::__set_exec_started()
{
    _exec_start = std::chrono::high_resolution_clock::now();
}

long FastRRT::__get_exec_time_ms()
{
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - _exec_start);
    return duration_ms.count();
}

bool FastRRT::__check_timeout()
{
    return (_timeout_ms > 0 && __get_exec_time_ms() > _timeout_ms);
}

void FastRRT::setPlanData(SearchParams &params)
{
    auto frame = params.frame();
    this->_hasPlanData = true;
    this->_start = params.start();
    this->_goal = params.goal();
    this->_ptr = frame->getCudaPtr();
    this->_planningVelocity_m_s = params.velocity_m_s();
    this->_timeout_ms = params.timeout_ms();
    this->_maxPathSize = params.maxPathSize_px();
    this->_distToGoalTolerance = params.distanceToGoalTolerance_px();
    this->_headingErrorTolerance = params.headingErrorTolerance();

    _graph.setSearchParams(params.minDistance(), _egoParams.egoLowerBound(), _egoParams.egoUpperBound());
    _graph.setClassCosts(frame->getCudaClassCostsPtr(), frame->getClassCount());
    _graph.processDirectGoalConnection(frame, _goal.x(), _goal.z(), _goal.heading(), 0.8);
    // printf ("_goal.x = %d, _goal.y = %d, _goal.h = %f\n", _goal.x(), _goal.z(), _goal.heading().deg());
}

// extern void exportGraph2(CudaGraph *graph, const char *filename);

void FastRRT::search_init(bool copyIntrinsicCostsFromFrame)
{
    if (!_hasPlanData)
    {
        throw std::runtime_error("unable to initialize planning without planning data");
    }
    __set_exec_started();
    _graph.clear();
    _graph.addStart(_start.x(), _start.z(), _start.heading());
    _last_expanded_node_count = 0;

    // int x = 183, z = 72;
    // printf ("result for %d,%d: z = %.2f\n", x, z, this->_ptr[z * 256 + x].z);
}

void FastRRT::__shrink_search_graph()
{
    std::vector<Waypoint> path = getPlannedPath();
    _graph.clear();
    for (Waypoint &p : path)
        _graph.setType(p.x(), p.z(), GRAPH_TYPE_NODE);
}

bool FastRRT::loop(bool smart)
{
    if (__check_timeout())
    {
        printf("timeout\n");
        return false;
    }

    bool expandFrontier = _last_expanded_node_count >= 100;

    // printf ("_last_expanded_node_count = %d\n", _last_expanded_node_count);

    //_graph.dumpNodesToFile("before_error_1.txt");
    if (smart)
    {
        _graph.smartExpansion(_ptr, _goal.heading(), _maxPathSize, _planningVelocity_m_s, expandFrontier, _last_expanded_node_count == 0, {_goal.x(), _goal.z()}, _goal.heading());
    }
    else
    {
        _graph.expandTree(_ptr, _goal.heading(), _maxPathSize, _planningVelocity_m_s, expandFrontier, {_start.x(), _start.z()}, {_goal.x(), _goal.z()}, _goal.heading());
    }

    _last_expanded_node_count = _graph.count(GRAPH_TYPE_TEMP);

    if (_last_expanded_node_count == 0)
    {
        if (_graph.countAll() == 0)
        {
            _graph.addStart(_start.x(), _start.z(), _start.heading());
            return true;
        }
    }

    //_graph.dumpNodesToFile("before_error_2.txt");
    _graph.acceptDerivedNodes({_goal.x(), _goal.z()}, _goal.heading().rad());

    // TODO: link last option to searchframe state
    if (_graph.findBestGoalDirectConnection(_ptr, _distToGoalTolerance, true))
    {
        float4 parent_in_graph = _graph.bestGraphDirectConnectionParent();
        // child
        float4 child_in_expansion_candidates = _graph.bestGraphDirectConnectionChild();

        const int parent_x = TO_INT(parent_in_graph.x);
        const int parent_z = TO_INT(parent_in_graph.y);
        const float parent_base_cost = _graph.getCost(parent_x, parent_z);

        _graph.add(TO_INT(child_in_expansion_candidates.x), TO_INT(child_in_expansion_candidates.y),
                   angle::rad(child_in_expansion_candidates.z),
                   parent_x, parent_z, parent_in_graph.w + parent_base_cost);

        const int child_x = TO_INT(child_in_expansion_candidates.x);
        const int child_z = TO_INT(child_in_expansion_candidates.y);
        const float child_cost = child_in_expansion_candidates.z + parent_base_cost;

        _graph.add(_goal.x(), _goal.z(), _goal.heading(), child_x, child_z, child_cost);

        // printf("[Direct connection] %d, %d --> %d, %d --> %d, %d with cost %f\n",
        //        parent_x, parent_z, child_x, child_z, _goal.x(), _goal.z(), child_cost);
    }

    // if (!_graph.checkGraphIsConsistent()) {
    //     //_graph.dumpNodesToFile("error.txt");
    //     printf ("[FAST-RRT ERROR] The graph is not a DAG anymore\n");
    //     return false;
    // }
    if (goalReached())
    {
        // printf ("shrinking graph...\n");
        //__shrink_search_graph();
        return false;
    }
    return true;
}

bool FastRRT::path_optimize()
{
    if (__check_timeout())
        return false;

    std::vector<Waypoint> res = getPlannedPath();

    sptr<float4> path = _graph.convertPlannedPath(res);

    //printf("[path optimize] size = %ld\n", res.size());

    // TODO: check if the distances are trully checked (last bool)
    return _graph.optimizePathLoop(_ptr, path, res.size(), _distToGoalTolerance, true);
}

bool FastRRT::goalReached()
{
    int2 goal = {_goal.x(), _goal.z()};
    return _graph.checkGoalReached(_ptr, goal, _goal.heading(), _distToGoalTolerance, _headingErrorTolerance.rad());
}

std::vector<Waypoint> FastRRT::getPlannedPath()
{
    std::vector<Waypoint> res;

    if (!_hasPlanData)
        return res;

    if (!goalReached())
        return res;

    int2 n = _graph.findBestNode(_ptr, _goal.heading(), _distToGoalTolerance, _goal.x(), _goal.z(), _headingErrorTolerance.rad());
    long i = 0;

    while (n.x != -1 && n.y != -1)
    {
        res.push_back(Waypoint(n.x, n.y, _graph.getHeading(n.x, n.y)));
        n = _graph.getParent(n.x, n.y);

        if (i++ >= 1000000)
        {
            printf("[ERROR] looping too much (%d, %d) i = %ld\n", n.x, n.y, i);
            res.clear();
            return res;
        }
    }

    std::reverse(res.begin(), res.end());
    return res;
}

extern std::vector<Waypoint> interpolate(std::vector<Waypoint> &path, int width, int height);

std::vector<Waypoint> FastRRT::interpolatePlannedPath()
{
    auto v = getPlannedPath();
    return interpolate(v, _graph.width(), _graph.height());
}

std::vector<Waypoint> FastRRT::interpolatePlannedPath(std::vector<Waypoint> path)
{
    return interpolate(path, _graph.width(), _graph.height());
}

std::vector<GraphNode> FastRRT::exportGraphNodes()
{
    std::vector<int3> nodes = _graph.listAll();
    std::vector<GraphNode> res;
    res.reserve(nodes.size());

    for (int3 n : nodes)
    {
        GraphNode g(n.x, n.y, n.z);
        int2 parent = _graph.getParent(n.x, n.y);
        g.parent_x = parent.x;
        g.parent_z = parent.y;
        g.heading_rad = _graph.getHeading(n.x, n.y).rad();
        g.cost = _graph.getCost(n.x, n.y);
        g.connectToEndCost = _graph.getDirectCost(n.x, n.z);
        res.push_back(g);
    }

    return res;
}

extern std::vector<Waypoint> interpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2);

std::vector<Waypoint> FastRRT::idealGeometryCurveNoObstacles(Waypoint goal)
{
    float3 *start = _graph.getCoordinateStart();
    return interpolateHermiteCurve(
        _graph.width(),
        _graph.height(),
        Waypoint(
            static_cast<int>((*start).x),
            static_cast<int>((*start).y),
            angle::rad(static_cast<float>((*start).z))),
        goal);
}

void FastRRT::computeGraphRegionDensity()
{
    _graph.computeGraphRegionDensity();
}

void FastRRT::saveCurrentGraphState(std::string filename)
{
    _graph.dumpGraph(filename.c_str());
}

void FastRRT::loadGraphState(std::string filename)
{
    _graph.readfromDump(filename.c_str());
    _last_expanded_node_count = _graph.count(GRAPH_TYPE_TEMP);
}
