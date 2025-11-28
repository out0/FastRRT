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
#TIMEOUT = 40000
TIMEOUT = -1


GRAPH_TYPE_NODE = 1
GRAPH_TYPE_TEMP = 2
GRAPH_TYPE_PROCESSING = 3


def log_graph(rrt: FastRRT, frame: CudaFrame, file: str) -> None:
    cimg = frame.get_color_frame()
    nodes = rrt.export_graph_nodes()
        
    width = cimg.shape[1]
    height = cimg.shape[0]
        
    for n in nodes:
        x = n[0]
        z = n[1]
        if (x < 0 or x > width): continue
        if (z < 0 or z > height): continue
        
        if int(n[2]) == GRAPH_TYPE_NODE:
            cimg[z, x, :] = [255, 255, 255]
        elif int(n[2]) == GRAPH_TYPE_PROCESSING:
            cimg[z, x, :] = [0, 255, 0]
        elif int(n[2]) == GRAPH_TYPE_TEMP:
            cimg[z, x, :] = [0, 0, 255]     
        
    cv2.imwrite(file, cimg)
    
   
def output_path_result(source_bev: str, path: np.ndarray, output: str) -> None:
    
    raw = np.array(cv2.imread(source_bev, cv2.IMREAD_COLOR), dtype=np.float32)
    
    frame = CudaFrame(
            frame=raw,
            lower_bound=Waypoint(119, 148),
            upper_bound=Waypoint(137, 108),
            min_dist_x=22,
            min_dist_z=40
        )
    
    f = frame.get_color_frame()
    
    for i in range(path.shape[0]):
        x = int(path[i,0])
        z = int(path[i,1])
        f[z, x, :] = [255, 255 , 255]
    
    cv2.imwrite(output, f)
    
def output_to_file(path: np.ndarray, output: str) -> None:
    
    with open(output, "w") as f:
        for i in range(path.shape[0]):
            x = int(path[i,0])
            z = int(path[i,1])            
            f.write(f"{x},{z},{path[i,2]}\n")
        

def measure_execution_time(func):
    start_time = time.time()  # Start the timer
    func()  # Call the function
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time  # Calculate the time taken
    print(f"Execution Time: {execution_time:.6f} seconds")

class TestFastRRT(unittest.TestCase):
    
    def test_search(self):
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
                
        bev = "/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/fast_rrt/tests/bev_1.png"
                
        raw = np.array(cv2.imread(bev, cv2.IMREAD_COLOR), dtype=np.float32)

        # for i in range(10):
        #     print (f"  {raw[3*i]}, {raw[3*i+1]}, {raw[3*i+2]}", end="");
        # print("")

        frame = CudaFrame(
            frame=raw,
            lower_bound=Waypoint(119, 148),
            upper_bound=Waypoint(137, 108),
            min_dist_x=22,
            min_dist_z=40
        )
 
        path = rrt.get_planned_path()
        self.assertTrue(path is None)

        ptr = frame.get_cuda_frame()

        rrt.set_plan_data(ptr, (128, 128, 0), (128, 0, 0), 1)
        
        min_exec_time = 9999999
        max_exec_time = -1        
        
        loop = True
        while loop:
            start_time = time.time()
            rrt.search_init()
            while (rrt.loop(False)):
                #log_graph(rrt, frame, "output1.png")
                pass
            
            end_time = time.time()

            if not rrt.goal_reached():
                continue
            #self.assertTrue(rrt.goal_reached())
            #print(f"goal reached? {rrt.goal_reached()}")

            execution_time_coarse = end_time - start_time  # Calculate the time taken
            #print(f"Coarse path: {1000*execution_time:.6f} ms")
            
            # path = rrt.get_planned_path(interpolate=True)
            # if path is None:
            #     #print("should be able to interpolate")
            #     continue
                
            start_time = time.time()
            for _ in range(10):
                rrt.loop_optimize()
            
            end_time = time.time()
            execution_time_optim = end_time - start_time  # Calculate the time taken
            #print(f"Optimized path: {1000*execution_time:.6f} ms")
            
            path = rrt.get_planned_path(interpolate=True)
            output_path_result(bev, path, "output1.png")
            
            if max_exec_time < 0:
                max_exec_time = execution_time_optim + execution_time_coarse
                min_exec_time = max_exec_time
                print(f"exec time (min): {1000*min_exec_time:.6f} ms")
                print(f"exec time (max): {1000*max_exec_time:.6f} ms\n\n")
            
            if min_exec_time > execution_time_optim + execution_time_coarse:
                min_exec_time = execution_time_optim + execution_time_coarse
                print(f"exec time (min): {1000*min_exec_time:.6f} ms")
                print(f"exec time (max): {1000*max_exec_time:.6f} ms\n\n")

            if max_exec_time < execution_time_optim + execution_time_coarse:
                max_exec_time = execution_time_optim + execution_time_coarse
                print(f"exec time (min): {1000*min_exec_time:.6f} ms")
                print(f"exec time (max): {1000*max_exec_time:.6f} ms\n\n")


            #output_to_file(rrt.get_planned_path(), "output1.txt")
            #output_to_file(path, "output1i.txt")
            loop = True
            

        i = 1


if __name__ == "__main__":
    unittest.main()