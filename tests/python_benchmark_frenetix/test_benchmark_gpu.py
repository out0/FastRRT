import unittest
from pyfastrrt import FastRRT
from pydriveless import SearchFrame, angle, EgoParams, SearchParams, Waypoint
import time, math
import cv2, numpy as np
from test_utils import TestUtils

MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 60000
TIMEOUT = -1

PERCEPTION_WIDTH_M = 1
PERCEPTION_HEIGHT_M = 1

class TestFastRRTFrenetix(unittest.TestCase):
    
    def test_execute_scenario(self):
        
        img = np.array(cv2.imread("converted_bev_23.png"), dtype=np.float32)
        start=Waypoint(416, 686, angle.new_deg(90 + -0.039754376))
        goal=Waypoint(296, 15, angle.new_deg(-14))
        
        TestUtils.print_distinc_classes(img)

        ego_params = EgoParams.init(img.shape[1], img.shape[0])\
            .with_segmentation_class_costs(np.array([-1, 0, 0, 0, 0]))\
            .with_segmentation_class_colors(np.array([
                [0, 0, 0],
                [128, 128, 128],
                [0, 0, 255],
                [255, 255, 255],
                [255, 0, 0]
            ]))\
            .with_ego_lower_bound((-1, -1))\
            .with_ego_upper_bound((-1, -1))\
            .with_max_steering_angle(angle.new_deg(40))\
            .with_max_curvature(0.34)\
            .with_search_physical_size(PERCEPTION_WIDTH_M, PERCEPTION_HEIGHT_M)\
            .with_vehicle_length(VEHICLE_LENGTH_M)\
            .build()

        # Create search frame from the ego parameters, which automatically configures the frame properties accordingly
        frame = ego_params.new_search_frame()
        # frame data copy from the image
        frame.set_frame_data(img)

        search_params = ego_params.new_search_params(
            start=Waypoint(416, 686, angle.new_deg(90 + -0.039754376)),
            goal=Waypoint(296, 15, angle.new_deg(-14)))\
            .with_distance_to_goal_tolerance(20.0)\
            .with_frame(frame)\
            .with_max_path_size(40.0)\
            .with_min_distance((2, 2))\
            .with_velocity(1.0)\
            .build()       
        
        rrt = FastRRT(ego_params)
        rrt.set_plan_data(search_params)
        
        
        start_time = time.time()
        frame.process_safe_distance_zone(min_distance=(2,2), compute_vectorized=False)
        frame.process_distance_to_goal(296, 15)
        rrt.search_init(True)
        loop_count = 1
        
        while not rrt.goal_reached() and rrt.loop(True):
            loop_count += 1
            #rrt.save_current_graph_state(f"log/loop_{loop_count}.dat")
            #nodes = rrt.export_graph_nodes()
            #TestUtils.output_path_result(frame, nodes, "output1.png", goal)

        end_time = time.time()
        execution_time = end_time - start_time
        
        path = rrt.get_planned_path(interpolate=False)
        if path is None:
            print(f"no path found")
            return False
        
        #np.save('coarse_path.npy', path)
        #rrt.save_current_graph_state("coarse_path_state.dat")
        
        path = rrt.get_planned_path(interpolate=True)
        print (f"found path with {len(path)} waypoints in {1000*execution_time:.2f} ms, took {loop_count} iterations")
        TestUtils.output_path_result(frame, path, "output1.png", (goal.x, goal.z, goal.heading.rad()))
        
        loop_count = 1
        start_time = time.time()
        while rrt.path_optimize():
            loop_count += 1
            pass
        end_time = time.time()
        execution_time = end_time - start_time

        path = rrt.get_planned_path(interpolate=True)
        print (f"optimizing path with {len(path)} waypoints in {1000*execution_time:.2f} ms took {loop_count} iterations")
        TestUtils.output_path_result(frame, path, "output1_optim.png", (goal.x, goal.z, goal.heading.rad()))

    

if __name__ == "__main__":
    unittest.main()