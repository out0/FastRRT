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
#TIMEOUT = -1
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
        
        data = TestUtils.timed_exec(lambda  :TestFrame(f"scenarios/{scenario.file}.pfm").get_data_cuda(cost_map=True, start=scenario.custom_start, goal=scenario.custom_goal))

        proc = TestUtils.pre_process_gpu(data, data.frame, MAX_STEERING_ANGLE, VEHICLE_LENGTH_M, copy_intrinsic_costs_from_frame=True)

        rrt = RRT(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=10,
            min_dist_z=10,
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
        
        S = Waypoint(start[0], start[1], start[2])
        G = Waypoint(goal[0], goal[1], goal[2])

        rrt.set_plan_data(
            proc,
            start=S,
            goal=G,
            velocity_m_s=1.0
        )
        
        
        start_time = time.time()
        rrt.search_init(MIN_DIST_GPU)
        loop_count = 0
        while not rrt.goal_reached() and rrt.loop_rrt_star(True):
            # partial_path = rrt.list_nodes()
            # if len(partial_path) > 0:
            #     path = convert_to_ndarray(partial_path)
            #     TestUtils.output_path_result_cpu(data.frame, path, f"output1.png")
            loop_count += 1       
        end_time = time.time()
        execution_time = end_time - start_time
        
        path = rrt.get_planned_path(interpolate=True)
        if path is None:
            print(f"[{scenario.file}] no path found")
            return False
        
        path = convert_to_ndarray(path)

        coarse_data = CurveAssessment(data.width(), data.height()).assess_curve(data.cpu_frame, path, start_heading=start[2], compute_heading=False)
        coarse_data.num_loops = loop_count
        coarse_data.proc_time_ms = execution_time * 1000
        coarse_data.timeout = TIMEOUT > 0 and coarse_data.proc_time_ms >= TIMEOUT
        coarse_data.goal_reached = rrt.goal_reached()
        coarse_data.coarse = True
        coarse_data.name = f"cpu_{scenario.file}"
        coarse_data.curve = convert_to_ndarray(rrt.get_planned_path(interpolate=False))
  
        print(f"[{scenario.file}] coarse path total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
        TestUtils.output_path_result_cpu(proc, path, f"test_result/coarse_{coarse_data.name}.png")       
        
        optim_data = None
        
        if optim_loop_count > 0 and not rrt.timeout():
            start_time = time.time()
            loop_count = 0
            while loop_count < optim_loop_count and rrt.loop_rrt_star(False):
                loop_count += 1
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[{scenario.file}] optim path total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
            
            path = convert_to_ndarray(rrt.get_planned_path(interpolate=True))
            
            optim_data = CurveAssessment(data.width(), data.height()).assess_curve(data.cpu_frame, path, start_heading=start[2], compute_heading=False)
            optim_data.num_loops = loop_count
            optim_data.proc_time_ms = execution_time * 1000
            optim_data.timeout = TIMEOUT > 0 and optim_data.proc_time_ms >= TIMEOUT
            optim_data.goal_reached = rrt.goal_reached()
            optim_data.coarse = False
            optim_data.name = f"cpu_{scenario.file}"
            
            TestUtils.output_path_result_cpu(proc, path, f"test_result/optim_{coarse_data.name}.png")
            optim_data.curve = convert_to_ndarray(rrt.get_planned_path(interpolate=False))
        
        result_file = f"test_result/results.csv"
        data_result_file = f"test_result/data_results.csv"
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write(coarse_data.to_csv_header())
                
        with open(result_file, "a") as f:
            f.write(coarse_data.to_csv())
            if optim_data is not None:
                f.write(optim_data.to_csv())
            
        with open(data_result_file, "a") as f:
            f.write(coarse_data.to_json())
            if optim_data is not None:
                f.write(optim_data.to_json())
        

    def test_cpu_scenarios(self):

        self.execute_scenario(TestScenario("map_cost_5",
                                           custom_start=(489, 770, math.radians(-45)),
                                           custom_goal=(428, 338, math.radians(45))), smart=True)

        self.execute_scenario(TestScenario("map_cost_8",
                                           custom_start=(243, 790, math.radians(-10)),
                                           custom_goal=(442, 450, math.radians(160))), smart=True)

        self.execute_scenario(TestScenario("map_cost_18",
                                           custom_start=(327, 223, math.radians(180)),
                                           custom_goal=(178, 534, math.radians(180))), smart=True)

        self.execute_scenario(TestScenario("map_cost_25",
                                           custom_start=(344, 428, math.radians(45)),
                                           custom_goal=(714, 528, math.radians(180))), smart=True)



        self.execute_scenario(TestScenario("map_cost_31",
                                           custom_start=(387, 416, math.radians(0)),
                                           custom_goal=(146, 264, math.radians(-170))), smart=True)


        self.execute_scenario(TestScenario("map_cost_39",
                                           custom_start=(389, 473, math.radians(0)),
                                           custom_goal=(781, 488, math.radians(180))), smart=True)


if __name__ == "__main__":
    unittest.main()