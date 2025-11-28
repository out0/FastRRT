import sys, time
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive")
import unittest, math
from fast_rrt.graph import CudaGraph
import numpy as np
from cudac.cuda_frame import CudaFrame, Waypoint
import cv2
import time
from test_utils import TestFrame, TestData, TestUtils

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 40000
#TIMEOUT = -1


GRAPH_TYPE_NODE = 1
GRAPH_TYPE_TEMP = 2
GRAPH_TYPE_PROCESSING = 3


def plot_costs (costs: np.ndarray, file: str) -> None:
    normalized = (costs / 10.0) * 255  # Scale 0-10.0 to 0-255
    img = normalized.astype(np.uint8)       # Convert to uint8 for cv2

# Save as grayscale PNG
    cv2.imwrite(file, img)
    pass



class TestCudaGraph(unittest.TestCase):
          
    
    def test_exec(self):
        raw = np.array(cv2.imread("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/fast_rrt/tests/bev_1.png", cv2.IMREAD_COLOR), dtype=np.float32)
         
        frame = CudaFrame(
            frame=raw,
            lower_bound=Waypoint(119, 148),
            upper_bound=Waypoint(137, 108),
            min_dist_x=0,
            min_dist_z=0
        )
        
        f = frame.get_color_frame()
        
        gray_frame = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        cv2.imwrite("empty.png", gray_frame)
        
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
            libdir=None
        )
        
        graph.compute_boundaries(frame)
        
        frame.invalidate_cpu_frame()
        p = frame.get_frame()
        
        for i in range (p.shape[0]):
            for j in range (p.shape[1]):
                if p[i, j, 2] == 0:
                    f[i, j, :] = [255, 255, 255]
                else:
                    f[i, j, :] = [0,0,0]
        
        cv2.imwrite("transf.png", f)



if __name__ == "__main__":
    unittest.main()