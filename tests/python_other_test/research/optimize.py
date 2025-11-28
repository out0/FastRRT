import sys, time
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive")
import unittest, math
from fast_rrt.fastrrt import FastRRT
import numpy as np
from cudac.cuda_frame import CudaFrame, Waypoint
import cv2

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 40000

def read_path(file: str) -> np.ndarray:
    with open(file, "r") as f:
        lines = f.readlines()
        path = np.zeros((len(lines), 3), dtype=np.float32)
        i = 0
        for l in lines:
            p = l.split(",")
            path[i, 0] = float(p[0])
            path[i, 1] = float(p[1])
            path[i, 2] = float(p[2])
            i += 1
        
        return path

def hermite_curve(p1, theta1, p2, theta2):
    x1, y1 = p1
    x2, y2 = p2

    num_points = 2 * abs(int(y2 - y1))

    # Tangents (scaled unit vectors)
    d = np.hypot(x2 - x1, y2 - y1)  # use distance to scale tangents
    t1 = d * np.array([np.cos(theta1), np.sin(theta1)])
    t2 = d * np.array([np.cos(theta2), np.sin(theta2)])

    t = np.linspace(0, 1, num_points)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    curve = np.outer(h00, p1) + np.outer(h10, t1) + np.outer(h01, p2) + np.outer(h11, t2)
    
    ideal = np.zeros((256), dtype=np.int32)
    
    for c in curve:
        x = int(c[0])
        y = int(c[1])
        ideal[y] = x
    
    return ideal

def optimize_loop(ideal: np.ndarray, curr: np.ndarray):
    
    for i in range(1, len(curr)):
        x = int(curr[i, 0])
        y = int(curr[i, 1])
        xi = ideal[y]
        
        if abs(x - xi) == 1:
            curr[i, 0] = xi    
        else:
            curr[i, 0] = int(0.5*(x + xi))
    

def measure_execution_time(func):
    start_time = time.time()  # Start the timer
    func()  # Call the function
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time  # Calculate the time taken
    print(f"Execution Time: {1000*execution_time:.6f} ms")

def print_path(f, path, color = [0, 0, 255]):
    for p in path:
        x = int(p[0])
        z = int(p[1])
        if (x < 0 or x > f.shape[1]): continue
        if (z < 0 or z > f.shape[0]): continue
        f[z, x, :] = color
    cv2.imwrite("tst.png", f)

class TestOptimze(unittest.TestCase):
    
    def test_optimize(self):
        path = read_path("output1.txt")
        n = len(path) - 1
        p1 = (path[0,0], path[0,1])
        p2 = (path[n,0], path[n,1])
        theta1 = path[0,2]
        theta2 = path[n,2]
        
        measure_execution_time(lambda: hermite_curve(p1, theta1, p2, theta2))
        
        ideal_points = hermite_curve(p1, theta1, p2, theta2)
        
        bev = "/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/fast_rrt/tests/bev_1.png"
        raw = np.array(cv2.imread(bev, cv2.IMREAD_COLOR), dtype=np.float32)
        
        frame = CudaFrame(
            frame=raw,
            lower_bound=Waypoint(119, 148),
            upper_bound=Waypoint(137, 108),
            min_dist_x=22,
            min_dist_z=40
        )
        
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
        
        f = frame.get_color_frame()
       
        # cv2.imwrite("tst.png", f)
       
        curve = rrt.ideal_curve(113, 0, math.radians(12))
        ideal = np.zeros((256), dtype=np.int32)
    
        for c in curve:
            x = int(c[0])
            y = int(c[1])
            if y > 128:
                p = 1
            ideal[y] = x
            f[y, x, :] = [255, 255 , 255]
                   
        print_path (f, path)
        
        
        i = 0
        while i < 10:
            optimize_loop(ideal_points, path)
            i += 1
        
        
        
        new_path = rrt.interpolate_planned_path_p(path)
            
        print_path (f, new_path, color=[255, 0, 0])
        
        
        
        # for p in curve:
        #     x = int(p[0])
        #     z = int(p[1])
        #     if (x < 0 or x > raw.shape[1]): continue
        #     if (z < 0 or z > raw.shape[0]): continue
        #     f[z, x, :] = [255, 255 , 255]
            
        # for p in path:
        #     x = int(p[0])
        #     z = int(p[1])
        #     if (x < 0 or x > raw.shape[1]): continue
        #     if (z < 0 or z > raw.shape[0]): continue
        #     f[z, x, :] = [0, 0 , 255]

        # cv2.imwrite("tst.png", f)

if __name__ == "__main__":
    unittest.main()