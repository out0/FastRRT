import sys, os
sys.path.append("../../")
sys.path.append("../../../")
import unittest
from fast_rrt.cpu_rrt import *
from test_utils import TestFrame, TestData, TestUtils
from fast_rrt.graph import CudaFrame, CudaGraph
import time
import numpy as np
from curve_quality import CurveAssessment

MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 60000
TIMEOUT = -1
SEGMENTATION_COST = [-1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1]

def get_test_data(file: str) -> TestData:
    return TestFrame(file).get_data_cuda()

class TestScenario:
    file: str
    custom_start: tuple[int, int, float]
    custom_goal: tuple[int, int, float]
    custom_start_heading: float
    custom_goal_heading: float

    def __init__(self, 
                 file: str, 
                 custom_start_heading: float = None,
                 custom_goal_heading: float = None,
                 custom_start: tuple[int, int, float] = None,
                 custom_goal: tuple[int, int, float] = None):
        self.file = file
        self.custom_start = custom_start
        self.custom_goal = custom_goal
        self.custom_start_heading = custom_start_heading
        self.custom_goal_heading = custom_goal_heading  

def convert_to_ndarray(path: list[tuple[int, int, float]]) -> np.ndarray:
    if path is None:
        return np.zeros((len(1), 3), dtype=np.float32)
    """
    Convert a list of tuples to a numpy array.
    """
    path_array = np.zeros((len(path), 3), dtype=np.float32)
    for i in range(len(path)):
        if len(path[i]) == 3:
            x, z, h = path[i]
        else:
            x, z = path[i]
            h = 0.0
        path_array[i, 0] = x
        path_array[i, 1] = z
        path_array[i, 2] = h
    return path_array


class TestFastRRT(unittest.TestCase):


    
    def execute_scenario(self, scenario: TestScenario, smart: bool = True, path_step_size: float = 50.0, dist_to_goal_tolerance: float = 15.0, optim_loop_count: int = 20):
        
        print (f"executing scenario {scenario.file}")
        
        f = TestFrame(f"test_scenarios/{scenario.file}.png")
        data: TestData = TestUtils.timed_exec(f.get_data_cpu)
        data_gpu: TestData = TestUtils.timed_exec(f.get_data_cuda)
        proc = TestUtils.pre_process_gpu(data, data_gpu.frame, MAX_STEERING_ANGLE, VEHICLE_LENGTH_M)

        rrt = RRT(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            # min_dist_x=22,
            # min_dist_z=40,
            min_dist_x=0,
            min_dist_z=0,
            lower_bound_x=-1,
            lower_bound_z=-1,
            upper_bound_x=-1,
            upper_bound_z=-1,
            max_path_size_px=path_step_size,
            dist_to_goal_tolerance_px=dist_to_goal_tolerance,
            class_cost=SEGMENTATION_COST
        )
        
        start = scenario.custom_start if scenario.custom_start else (data.start.x, data.start.z, 0.0)
        goal = scenario.custom_goal if scenario.custom_goal else (data.goal.x, data.goal.z, 0.0)
        
        if scenario.custom_start_heading is not None:
            start = (start[0], start[1], scenario.custom_start_heading)
        if scenario.custom_goal_heading is not None:
            goal = (goal[0], goal[1], scenario.custom_goal_heading)
        
        rrt.set_plan_data(
            proc,
            start=Waypoint(start[0], start[1], start[2]),
            goal=Waypoint(goal[0], goal[1], goal[2]),
            velocity_m_s=1.0
        )
        
        
        start_time = time.time()
        rrt.search_init(MIN_DIST_GPU)
        loop_count = 0
        while not rrt.goal_reached() and rrt.loop_rrt_star(False):
            partial_path = rrt.list_nodes()
            if len(partial_path) > 0:
                path = convert_to_ndarray(partial_path)
                TestUtils.output_path_result_cpu(data.frame, path, f"output1.png")
            loop_count += 1       
        end_time = time.time()
        execution_time = end_time - start_time
        
        path = rrt.get_planned_path(interpolate=True)
        if path is None:
            print(f"[{scenario.file}] no path found")
            return False
        
        path = convert_to_ndarray(path)

        coarse_data = CurveAssessment(data.width(), data.height()).assess_curve(path, start_heading=start[2], compute_heading=False)
        coarse_data.num_loops = loop_count
        coarse_data.proc_time_ms = execution_time * 1000
        coarse_data.timeout = TIMEOUT > 0 and coarse_data.proc_time_ms >= TIMEOUT
        coarse_data.goal_reached = rrt.goal_reached()
        coarse_data.coarse = True
        coarse_data.name = f"gpu_{scenario.file}"
        coarse_data.curve = convert_to_ndarray(rrt.get_planned_path(interpolate=False))
  
        print(f"[{scenario.file}] coarse path total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
        TestUtils.output_path_result_cpu(data.frame, path, f"output1.png")       
        
        optim_loop_count = 2000
        optim_total_count = 40000
        count = 0
        
        
        while count < optim_total_count:
            start_time = time.time()
            loop_count = 0
            while loop_count < optim_loop_count and rrt.loop_rrt_star(True):
                loop_count += 1
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[{scenario.file}] optim path total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
                
            path = convert_to_ndarray(rrt.get_planned_path(interpolate=True))
            TestUtils.output_path_result_cpu(data.frame, path, f"output1.png")
            count += loop_count
        
        
        

    def test_cpu_scenarios(self):

        # self.execute_scenario(TestScenario("large_1"), path_step_size=100.0)

        # self.execute_scenario(TestScenario("large_2",
        #                                    custom_start_heading=math.radians(90), 
        #                                    custom_goal_heading=math.radians(45)))

        # self.execute_scenario(TestScenario("large_3",
        #                                    custom_start_heading=math.radians(90), 
        #                                    custom_goal_heading=math.radians(45)))

        #self.execute_scenario(TestScenario("small_1"), smart=False)

        self.execute_scenario(TestScenario("small_2"), smart=False, path_step_size=10.0, dist_to_goal_tolerance=15.0, optim_loop_count=2000)
        
        #self.execute_scenario(TestScenario("small_3"), smart=False, path_step_size=10.0, dist_to_goal_tolerance=15.0, optim_loop_count=20000)        


if __name__ == "__main__":
    unittest.main()