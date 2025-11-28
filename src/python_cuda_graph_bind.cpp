#include "../include/graph.h"
#include <driveless/search_frame.h>
#include "../include/fastrrt.h"

extern "C"
{
    void *cudagraph_initialize(
        int width,
        int height,
        float perceptionWidthSize_m,
        float perceptionHeightSize_m,
        float maxSteeringAngle_deg,
        float vehicleLength,
        int minDistance_x, int minDistance_z,
        int lowerBound_x, int lowerBound_z,
        int upperBound_x, int upperBound_z,
        float *segmentationClassCost, float max_curvature)
    {
        CudaGraph *g = new CudaGraph(width, height);
        g->setSearchParams({minDistance_x, minDistance_z},
                           {lowerBound_x, lowerBound_z},
                           {upperBound_x, upperBound_z});
        g->setPhysicalParams(perceptionWidthSize_m, perceptionHeightSize_m, angle::deg(maxSteeringAngle_deg), vehicleLength, max_curvature);

        std::vector<float> costs;
        int count = segmentationClassCost[0];
        costs.reserve(count);

        for (int i = 1; i <= count; i++)
            costs.push_back(segmentationClassCost[i]);

        g->setClassCosts(costs);
        return g;
    }
    void cudagraph_destroy(void *ptr)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        delete graph;
    }

    void compute_apf_repulsion(void *ptr, void *cudaFramePtr, float kr, int radius)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        SearchFrame *frame = (SearchFrame *)cudaFramePtr;
        graph->computeRepulsiveFieldAPF(frame->getCudaPtr(), kr, radius);
    }

    void compute_apf_attraction(void *ptr, void *cudaFramePtr, float ka, int goal_x, int goal_z)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        SearchFrame *frame = (SearchFrame *)cudaFramePtr;
        graph->computeAttractiveFieldAPF(frame->getCudaPtr(), ka, {goal_x, goal_z});
    }

    float *get_intrinsic_costs(void *ptr)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        float4 *frameData = graph->getFrameDataPtr()->getCudaPtr();

        int width = graph->width();
        int height = graph->height();

        float *data = new float[width * height];

        for (int h = 0; h < height; h++)
            for (int w = 0; w < width; w++)
            {
                int i = h * width + w;
                data[i] = frameData[i].z;
            }

        return data;
    }

    void destroy_intrinsic_costs_ptr(float *ptr)
    {
        delete[] ptr;
    }

    void add(void *ptr, int x, int z, float heading_rad, int parent_x, int parent_z, float cost)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        graph->add(x, z, angle::rad(heading_rad), parent_x, parent_z, cost);
    }

    void add_temporary(void *ptr, int x, int z, float heading_rad, int parent_x, int parent_z, float cost)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        graph->addTemporary(x, z, angle::rad(heading_rad), parent_x, parent_z, cost);
    }


    void derivate_node(void *ptr, void *searchFramePtr, float steering_angle, int path_size, float velocity, int x, int z)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        SearchFrame *f = (SearchFrame *)searchFramePtr;
        graph->derivateNode(f->getCudaPtr(), angle::rad(steering_angle), static_cast<double>(path_size), velocity, x, z);
    }

    void accept_derived_nodes(void *ptr, int goal_x, int goal_z, float goal_heading)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        graph->acceptDerivedNodes({goal_x, goal_z}, goal_heading);
    }
    bool check_in_graph(void *ptr, int x, int z)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        return graph->checkInGraph(x, z);
    }
    float get_heading(void *ptr, int x, int z)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        return graph->getHeading(x, z).rad();
    }
    void clear(void *ptr)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        graph->clear();
    }

    float *list_all(void *ptr)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        std::vector<int3> nodes = graph->listAll();

        float *res = new float[nodes.size() * 4 + 1];
        int pos = 1;

        res[0] = 0.0 + nodes.size();
        for (auto n : nodes)
        {
            res[pos] = static_cast<float>(n.x);
            res[pos + 1] = static_cast<float>(n.y);
            res[pos + 2] = graph->getHeading(n.x, n.y).rad();
            res[pos + 3] = n.z;
            pos += 4;
        }

        return res;
    }

    void free_list_all(float *p)
    {
        delete[] p;
    }

    bool can_connect_to_goal_using_hermite(void *ptr, void *frame_ptr, int x, int z, int goal_x, int goal_z, float goal_heading)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        SearchFrame *frame = (SearchFrame *)frame_ptr;

        return graph->canConnectToGoal(frame, x, z, goal_x, goal_z, goal_heading);
    }

    void solve_collisions(void *ptr)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        graph->solveCollisions();
    }

    void process_direct_goal_connection(void *ptr, void *frame_ptr, int goal_x, int goal_z, float goal_heading_rad, float max_curvature)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        SearchFrame *frame = (SearchFrame *)frame_ptr;

        graph->processDirectGoalConnection(frame, goal_x, goal_z, angle::rad(goal_heading_rad), max_curvature);
    }

    bool is_directly_connected_to_goal(void *ptr, int x, int z)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        return graph->isDirectlyConnectedToGoal(x, z);
    }

    float direct_connection_to_goal_cost(void *ptr, int x, int z)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        return graph->directConnectionToGoalCost(x, z);
    }

    float direct_connection_to_goal_heading(void *ptr, int x, int z)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        angle a = graph->directConnectionToGoalHeading(x, z);
        return a.rad();
    }

    void dump_graph_to_file (void *ptr, char *filename) {
        CudaGraph *graph = (CudaGraph *)ptr;
        graph->dumpGraph(filename);
    }

    void read_from_dump_file (void *ptr, char *filename) {
        CudaGraph *graph = (CudaGraph *)ptr;
        graph->readfromDump(filename);
        
    }

    int * get_parent(void *ptr, int x, int z) {
        CudaGraph *graph = (CudaGraph *)ptr;
        int2 p = graph->getParent(x, z);
        int *data_ptr = new int[2];
        data_ptr[0] = p.x;
        data_ptr[1] = p.y;
        return data_ptr;
    }
        
    void free_parent_data(int *data_ptr) {
        delete []data_ptr;
    }
        
    float get_cost(void *ptr, int x, int z) {
        CudaGraph *graph = (CudaGraph *)ptr;
        return graph->getCost(x, z);
    }

    int count_all(void *ptr) {
        CudaGraph *graph = (CudaGraph *)ptr;
        return graph->countAll();
    }

}