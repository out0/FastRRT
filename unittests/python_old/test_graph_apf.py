import sys, time
import unittest, math
from pyfastrrt import CudaGraph
import numpy as np
from pydriveless import SearchFrame, Waypoint
import cv2
import time
from test_utils import TestFrame, TestData, TestUtils, SEGMENTATION_CLASS_COST

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
#TIMEOUT = 40000
TIMEOUT = -1


GRAPH_TYPE_NODE = 1
GRAPH_TYPE_TEMP = 2
GRAPH_TYPE_PROCESSING = 3


def plot_costs (costs: np.ndarray, file: str) -> None:
    normalized = (costs / 10.0) * 255  # Scale 0-10.0 to 0-255
    img = normalized.astype(np.uint8)       # Convert to uint8 for cv2

# Save as grayscale PNG
    cv2.imwrite(file, img)
    pass

def measure_execution_time(func):
    start_time = time.time()  # Start the timer
    func()  # Call the function
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time  # Calculate the time taken
    print(f"Execution Time: {1000*execution_time:.6f} ms")


class TestCudaGraph(unittest.TestCase):
    
    def test_apf(self):
        graph = CudaGraph(
            width=256,
            height=256,
            perception_height_m=OG_REAL_HEIGHT,
            perception_width_m=OG_REAL_WIDTH,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=-1,
            lower_bound_z=-1,
            upper_bound_x=-1,
            upper_bound_z=-1,
            path_costs=SEGMENTATION_CLASS_COST
        )
        
        raw = np.full((256, 256, 3), 1.0, dtype=float)
        
        raw[128, 128, 0] = 0.0 # add obstacle in center pos
        
        img_ptr = SearchFrame(
            width=raw.shape[1],
            height=raw.shape[0],
            lower_bound=(119, 148),
            upper_bound=(137, 108),
        )
        img_ptr.set_frame_data(raw)
        graph.compute_apf_repulsion(img_ptr, 2.0, 50)
        
        costs = graph.get_intrinsic_costs()
        
        # for h in range(256):
            # for w in range(256):
            #     if costs[h, w] > 0:
            #         print(f"({w}, {h}) = {costs[h, w]}")
        
        plot_costs(costs, "test.png")

    def test_apf2(self):
        graph = CudaGraph(
            width=256,
            height=256,
            perception_height_m=OG_REAL_HEIGHT,
            perception_width_m=OG_REAL_WIDTH,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=119,
            lower_bound_z=148,
            upper_bound_x=137,
            upper_bound_z=108,
            path_costs=SEGMENTATION_CLASS_COST
        )
        
        raw = np.array(cv2.imread("bev_1.png", cv2.IMREAD_COLOR), dtype=np.float32)
        img_ptr = SearchFrame(
            width=raw.shape[1],
            height=raw.shape[0],
            lower_bound=(119, 148),
            upper_bound=(137, 108),
        )
        img_ptr.set_frame_data(raw)
        
        measure_execution_time(lambda  : graph.compute_apf_repulsion(img_ptr, 2.0, 50))
        
        costs = graph.get_intrinsic_costs()
        
        plot_costs(costs, "test2.png")
        
        measure_execution_time(lambda  : graph.compute_apf_attraction(img_ptr, 2.0, 128, 0))
        

if __name__ == "__main__":
    unittest.main()