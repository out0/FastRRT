#include "../include/fastrrt.h"
#include <driveless/search_frame.h>


extern "C"
{
    void *fastrrt_initialize(
        int width,
        int height,
        float perceptionWidthSize_m,
        float perceptionHeightSize_m,
        float maxSteeringAngle_deg,
        float vehicleLength,
        int lowerBound_x, int lowerBound_z,
        int upperBound_x, int upperBound_z,
        float *classCosts,
        float max_curvature)
    {

        EgoParams params = EgoParams::init(width, height)
            .withEgoLowerBound({lowerBound_x, lowerBound_z})
            .withEgoUpperBound({upperBound_x, upperBound_z})
            .withMaxCurvature(max_curvature)
            .withSearchPhysicalSize(perceptionWidthSize_m, perceptionHeightSize_m)
            .withMaxSteeringAngle(angle::deg(maxSteeringAngle_deg))
            .withVehicleLength(vehicleLength)
            .build();

        return new FastRRT(params);
    }

    void fastrrt_destroy(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        delete rrt;
    }

    void set_plan_data(void *ptr, void *cudaFramePtr, 
        int start_x, int start_z, float start_heading_rad, int goal_x, int goal_z, 
        float goal_heading_rad, float velocity_m_s, 
        int min_dist_x, int min_dist_z,
        int timeout_ms,
        float maxPathSize,
        float distToGoalTolerance,
        float headingErrorTolerance_rad)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        Waypoint s(start_x, start_z, angle::rad(start_heading_rad));
        Waypoint p(goal_x, goal_z, angle::rad(goal_heading_rad));
        // printf ("p.x = %d, p.y = %d, p.h = %f\n", p.x(), p.z(), p.heading().deg());
        
        SearchFrame *frame = (SearchFrame *)cudaFramePtr;
        auto params = SearchParams::init(s, p)
            .withVelocity(velocity_m_s)
            .withMinDistance({min_dist_x, min_dist_z})
            .withTimeout(timeout_ms)
            .withMaxPathSize(maxPathSize)
            .withDistanceToGoalTolerance(distToGoalTolerance)
            .withHeadingErrorTolerance(angle::rad(headingErrorTolerance_rad))
            .withFrame(frame)
            .build();

        rrt->setPlanData(params);
    }

    bool goal_reached(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->goalReached();
    }

    void search_init(void *ptr, bool copyIntrinsicCostsFromFrame)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->search_init(copyIntrinsicCostsFromFrame);
    }
    bool loop(void *ptr, bool smartExpansion)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->loop(smartExpansion);
    }

    bool path_optimize(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->path_optimize();
    }

    int *export_graph_nodes(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        auto nodes = rrt->exportGraphNodes();

        int *res = new int[3 * nodes.size() + 1];

        res[0] = nodes.size();

        int i = 1;
        for (auto n : nodes)
        {
            res[i] = n.x;
            res[i + 1] = n.z;
            res[i + 2] = n.nodeType;
            i += 3;
        }

        return res;
    }
    void release_export_graph_nodes(float *ptr)
    {
        delete[] ptr;
    }

    float *convertPath(std::vector<Waypoint> &path)
    {
        int size = path.size();

        // printf("size = %d\n", size);

        float *res = new float[3 * size + 1];
        res[0] = (float)size;

        // printf("res[0] = %f\n", res[0]);

        int i = 0;
        for (auto p : path)
        {
            int pos = (3 * i + 1);
            res[pos] = p.x();
            res[pos + 1] = p.z();
            res[pos + 2] = p.heading().rad();
            i += 1;
        }

        return res;
    }

    float *get_planned_path(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        std::vector<Waypoint> path = rrt->getPlannedPath();
        return convertPath(path);
    }

    float *interpolate_planned_path(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        std::vector<Waypoint> path = rrt->interpolatePlannedPath();
        return convertPath(path);
    }

    void release_planned_path_data(float *ptr)
    {
        delete[] ptr;
    }

    float *interpolate_planned_path_p(void *ptr, float *p, int size)
    {
        FastRRT *rrt = (FastRRT *)ptr;

        std::vector<Waypoint> pref;

        for (int i = 0; i < size; i += 3)
        {
            pref.push_back(Waypoint(p[i], p[i + 1], angle::rad(p[i + 2])));
            printf("(%d, %d, %f)\n", (int)p[i], (int)p[i + 1], p[i + 2]);
        }

        std::vector<Waypoint> path = rrt->interpolatePlannedPath(pref);
        return convertPath(path);
    }

    float *ideal_curve(void *ptr, int goal_x, int goal_z, float goal_heading_rad)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        std::vector<Waypoint> path = rrt->idealGeometryCurveNoObstacles({goal_x,
                                                                         goal_z,
                                                                         angle::rad(goal_heading_rad)});
        return convertPath(path);
    }

    void compute_region_debug_performance(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->computeGraphRegionDensity();
    }

    void save_current_graph_state(void *ptr, const char *filename)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->saveCurrentGraphState(std::string(filename));
    }

    void load_graph_state(void *ptr, const char *filename)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->loadGraphState(std::string(filename));
    };

}