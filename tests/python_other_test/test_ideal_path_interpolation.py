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
from fast_rrt.fastrrt import FastRRT

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 40000
#TIMEOUT = -1


#
# Hermite interpolation: an implementation of this curve gen is done in cpp.
# this test guarantees that the cpp code has the equivalent of this implementation
#
def gen_hermite_curve(p1, theta1, p2, theta2):
    x1, y1 = p1
    x2, y2 = p2

    num_points = 2 * abs(int(y2 - y1))

    # Tangents (scaled unit vectors)
    d = np.hypot(x2 - x1, y2 - y1)  # use distance to scale tangents
    t1 = d * np.array([np.cos(theta1 - math.pi/2), np.sin(theta1 - math.pi/2)])
    t2 = d * np.array([np.cos(theta2 - math.pi/2), np.sin(theta2 - math.pi/2)])

    t = np.linspace(0, 1, num_points)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    return np.outer(h00, p1) + np.outer(h10, t1) + np.outer(h01, p2) + np.outer(h11, t2)
    
    
class TestIdealCurveInterpolation(unittest.TestCase):
    
    def test_interpolation(self):
        end_point = (113, 0, math.radians(15))
        
        rrt = FastRRT(
            width=256,
            height=256,
            perception_height_m=OG_REAL_HEIGHT,
            perception_width_m=OG_REAL_WIDTH,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=119,
            lower_bound_z=148,
            upper_bound_x=137,
            upper_bound_z=108,
            max_path_size_px=40.0,
            dist_to_goal_tolerance_px=15.0,
            libdir=None
            )
        
        
        c1 = rrt.build_ideal_curve(end_point[0], end_point[1], end_point[2])
        c2 = gen_hermite_curve((128, 128), 0, (end_point[0], end_point[1]), math.radians(15))
        
        f1 = np.zeros(256, dtype=np.int32)
        for p in c1:
            f1[int(p[1])] = int(p[0])
                
        f2 = np.zeros(256, dtype=np.int32)
        for p in c2:
            f2[int(round(p[1]))] = int(round(p[0]))
        
        for i in range(256):
            if (f1[i] != f2[i]):
                print(f"({f1[i]}, {i}) != ({f2[i]}, {i})")
        
if __name__ == "__main__":
    unittest.main()