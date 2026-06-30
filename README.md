# FastRRT 

FastRRT is a RRT-based trajectory planning algorithm that generates hard-constrained smooth trajectories by performing controlled exponential nonholonomic tree expansion, path optimization, and collision check in parallel using many-core GPUs.

## Intro

We present a variation of RRT* that can explore and exploit the search space (SE) in parallel using many-core GPUs, efficiently producing an initial path that is collision-free, kinematically constrained, and smooth enough to be tracked by the vehicle. We divide the SE into sub-regions to synchronize exploration and exploitation based on node density. This allows the expansion to be exponential only in low-density areas, which corresponds to the exploitation task, opening several search paths for future exploration. In high-density areas, only leaf branches are expanded, which corresponds to
the exploration task. After each iteration, the number of newly added nodes is used to evaluate graph expansion and prevent overloading GPU cores, effectively controlling the number of parallel expansions, while still allowing new areas to be discovered.
\
Graph expansion resulting in collisions with existing branches is handled by efficiently erasing the colliding subtree to allow for graph reshaping, similar to RRT*, but preserving heading consistency. This reshaping is performed in parallel to achieve the desired efficiency. After the initial path
is defined, the coarse path is optimized using Hermite curve interpolation

## Reference

[Towards Real-time Nonholonomic Local Trajectory Planning Based on RRT using Many-core GPUs](https://www.techrxiv.org/doi/full/10.36227/techrxiv.175322833.30527176)


### Basic setup
```python 

ego_params = EgoParams(
    search_frame_dimensions=(800, 800),
    search_frame_physical_dimensions=(32.345, 32.345),
    ego_upper_bound=(375, 350),
    ego_lower_bound=(425, 400),
    max_curvature=1.432,
    max_steering_angle=40,
    meters_to_pixel_ratio_width=24.7334,
    meters_to_pixel_ratio_height=24.7334,
    pixel_to_meters_ratio_width=0.04041,
    pixel_to_meters_ratio_height=0.04041,
    segmentation_class_colors=np.array([
        (0, 0, 0),
        (255, 255, 255)
    ], dtype=np.int32),
    segmentation_class_costs=np.array([
        -1,
        1.2
    ], dtype=np.float32),
    vehicle_length_m=5.412658774,
    world_origin=WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))
)

search_params = SearchParams.init(
    start=Waypoint(375, 345, angle.new_deg(0)),
    goal=Waypoint(800, 112, angle.new_deg(0))
    ).with_world_origin(world_origin)\
    .with_distance_to_goal_tolerance(distance_px=15)\
    .with_velocity(velocity_m_s=1.0)\
    .with_map_origin(origin=MapPose(0, 0, 0, heading=angle.new_rad(0)))\
    .with_ego_pose(pose=MapPose(0, 0, 0, heading=conf.start.heading))\
    .with_heading_error_tolerance(angle.new_deg(5))\
    .with_timeout(timeout)\
    .with_max_path_size(40)\
    .with_min_distance((20, 20))\
    .with_frame(frame)\
    .build()

planner = FastRRT(ego_params=ego_params)
planner.set_plan_data(search_params)

planner.initialize()

while planner.planning_loop():
    pass

# Path Fast Optimization
while planner.path_optimize_loop():
    pass

path, cost  = planner.get_interpolated_planned_path()

``` 



