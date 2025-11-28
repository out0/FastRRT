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

planner = FastRRT(
    search_frame=cframe,
    perception_width_m=OG_REAL_WIDTH,
    perception_height_m=OG_REAL_HEIGHT,
    max_steering_angle_deg=MAX_STEERING_ANGLE,
    vehicle_length_m=VEHICLE_LENGTH_M,
    timeout_ms=100,
    min_dist_x=MIN_DISTANCE_WIDTH_PX,
    min_dist_z=MIN_DISTANCE_HEIGHT_PX,
    path_costs=SEGMENTATION_CLASS_COST
)
        
planner.set_plan_data(
    cuda_ptr=cframe,
    start=(128, 128, 0.0),
    goal=(128, 0, 0.0),
    velocity_m_s=velocity
)

planner.search_init()

while not planner.goal_reached():
    planner.loop(False)

# Path Fast Optimization
planner.path_optimize()

path = planner.get_planned_path(True)

``` 



