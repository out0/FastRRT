import sys, time
import unittest, math
import numpy as np
from test_utils import TestUtils, TestFrame, TestData
from cpu_rrt import RRT

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = -1
SEGMENTATION_COST = [-1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1]


def dev_frame() -> np.ndarray:
    #return np.full((1024, 1024, 3), 1.0, dtype=np.float32)
    return np.full((256, 256, 3), 1.0, dtype=np.float32)


def loop_rrt(rrt: RRT) -> bool:
    p = not rrt.goal_reached() and rrt.loop_rrt(True)
    
    return p
    
def test_rrt():
    
    # og_width = 256
    # og_height = 256
    
    # og_real_width = (OG_REAL_WIDTH / 256) * og_width
    # og_real_height = (OG_REAL_HEIGHT / 256) * og_width
    
    #img = TestUtils.timed_exec(dev_frame)
    
    data: TestData = TestUtils.timed_exec(lambda: TestFrame("custom3.png").get_data_cpu())
        
        
    TestUtils.output_path_result_cpu(data.frame, None, "output1.png")
    
    rrt = RRT(
        width=data.width(),
        height=data.height(),
        perception_height_m=data.real_height(),
        perception_width_m=data.real_width(),
        max_steering_angle_deg=MAX_STEERING_ANGLE,
        vehicle_length_m=VEHICLE_LENGTH_M,
        timeout_ms=TIMEOUT,
        min_dist_x=22,
        min_dist_z=40,
        lower_bound_x=119,
        lower_bound_z=148,
        upper_bound_x=137,
        upper_bound_z=108,
        max_path_size_px=30.0,
        dist_to_goal_tolerance_px=15.0,
        class_cost=SEGMENTATION_COST
        )
    
    rrt.set_plan_data(data.frame, data.start, data.goal, 1)

    loop = True
    while loop:
        TestUtils.timed_loop_exec("Coarse path", lambda: rrt.search_init(True), lambda: not rrt.goal_reached() and rrt.loop_rrt(True))
                    
        #self.assertTrue(rrt.goal_reached())
        print(f"goal reached? {rrt.goal_reached()}")
        
        path = rrt.get_planned_path(True)
        
        TestUtils.output_path_result_cpu(data.frame, path, "output1.png")
        
        
        TestUtils.timed_loop_exec_count("Optimized path", 30, None, rrt.loop_rrt, True)
        
        path = rrt.get_planned_path(True)
        TestUtils.output_path_result_cpu(data.frame, path, "output1.png")
        loop = False    
    



def test_rrt_star():
    
    # og_width = 256
    # og_height = 256
    
    # og_real_width = (OG_REAL_WIDTH / 256) * og_width
    # og_real_height = (OG_REAL_HEIGHT / 256) * og_width
    
    #img = TestUtils.timed_exec(dev_frame)
    
    data: TestData = TestUtils.timed_exec(lambda: TestFrame("custom3.png").get_data_cpu())
        
        
    TestUtils.output_path_result_cpu(data.frame, None, "output1.png")
    
    rrt = RRT(
        width=data.width(),
        height=data.height(),
        perception_height_m=data.real_height(),
        perception_width_m=data.real_width(),
        max_steering_angle_deg=MAX_STEERING_ANGLE,
        vehicle_length_m=VEHICLE_LENGTH_M,
        timeout_ms=TIMEOUT,
        min_dist_x=22,
        min_dist_z=40,
        lower_bound_x=119,
        lower_bound_z=148,
        upper_bound_x=137,
        upper_bound_z=108,
        max_path_size_px=30.0,
        dist_to_goal_tolerance_px=15.0,
        class_cost=SEGMENTATION_COST
        )
    
    rrt.set_plan_data(data.frame, data.start, data.goal, 1)

    loop = True
    while loop:
        TestUtils.timed_loop_exec("Coarse path", lambda: rrt.search_init(False), lambda: not rrt.goal_reached() and rrt.loop_rrt_star(True))
                    
        #self.assertTrue(rrt.goal_reached())
        print(f"goal reached? {rrt.goal_reached()}")
        
        path = rrt.get_planned_path(True)
        
        TestUtils.output_path_result_cpu(data.frame, path, "output1.png")
        
        
        TestUtils.timed_loop_exec_count("Optimized path", 30, None, rrt.loop_rrt_star, True)
        
        TestUtils.output_path_result_cpu(data.frame, path, "output1.png")
        loop = False    
    

    
if __name__ == "__main__":
    test_rrt()

