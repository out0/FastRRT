import unittest
from pyfastrrt import FastRRT
from pydriveless import SearchFrame, angle
import time, math
import cv2, numpy as np
from test_utils import TestUtils

MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 60000
TIMEOUT = -1

PERCEPTION_WIDTH_M = 1
PERCEPTION_HEIGHT_M = 1

class TestOptimization(unittest.TestCase):
    
    def test_execute_scenario(self):
        
        img = np.array(cv2.imread("converted_bev_23.png"), dtype=np.float32)
        
        frame = SearchFrame (
            width=img.shape[1],
            
            height=img.shape[0],
            lower_bound=(-1, -1),
            upper_bound=(-1, -1),
        )
    
        frame.set_class_costs(np.array([-1, 0, 0, 0, 0]))
        frame.set_class_colors((np.array([
            [0, 0, 0],
            [128, 128, 128],
            [0, 0, 255],
            [255, 255, 255],
            [255, 0, 0]
        ])))
        frame.set_frame_data(img)

        TestUtils.print_distinc_classes(img)

        
        rrt = FastRRT(
            search_frame=frame,
            perception_width_m=PERCEPTION_WIDTH_M,
            perception_height_m=PERCEPTION_HEIGHT_M,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            path_costs=np.array([-1, 0, 0, 0, 0], dtype=np.float32),
            max_path_size_px=40.0,
            dist_to_goal_tolerance_px=20.0,
            max_curvature=0.80
        )
        
        start = (416, 686, angle.new_deg(90 + -0.039754376).rad())
        goal = (296, 15, angle.new_deg(-14).rad())
        #goal = (296, 15, angle.new_deg(-8.9513413283239596).rad())
        
        rrt.set_plan_data(
            frame,
            start=start,
            goal=goal,
            velocity_m_s=1.0,
            min_dist=(2, 2)
        )
        
        
        start_time = time.time()

        frame.process_safe_distance_zone(min_distance=(2,2), compute_vectorized=False)
        frame.process_distance_to_goal(296, 15)


        rrt.search_init(True)
        rrt.load_graph_state("coarse_path_state.dat")

        print("Path: ")
        path = rrt.get_planned_path()
        first = True
        for p in path:
            if not first: print(" --> ", end="")
            print(f"{int(p[0]), int(p[1])}", end="")
            first = False
        print("\n")
       

        path = rrt.get_planned_path(interpolate=True)
#        print (f"found path with {len(path)} waypoints in {1000*execution_time:.2f} ms")
        TestUtils.output_path_result(frame, path, "output1.png", goal)

 
        start_time = time.time()
        for i in range(10):
            rrt.path_optimize()
        end_time = time.time()
        execution_time = end_time - start_time
        print (f"optimze path with {len(path)} waypoints in {1000*execution_time:.2f} ms")
            
        path = rrt.get_planned_path(interpolate=True)
        TestUtils.output_path_result(frame, path, "output1.png", goal)

    

if __name__ == "__main__":
    unittest.main()