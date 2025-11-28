import sys
sys.path.append("../../../")
sys.path.append("../../../../")
import cv2, numpy as np, json
import math
from utils.cudac.cuda_frame import CudaFrame
from model.physical_parameters import PhysicalParameters
from model.waypoint import Waypoint
from utils.fast_rrt.fastrrt import FastRRT
from utils.fast_rrt.graph import CudaGraph
import time

PROPORTION_meters_per_px = 0.135316469
GRAPH_TYPE_NODE = 1
GRAPH_TYPE_TEMP = 2
GRAPH_TYPE_PROCESSING = 3

class TestData:
    frame: CudaFrame
    start: Waypoint
    goal: Waypoint
    upper_bound: Waypoint
    lower_bound: Waypoint
    
    def __init__(self, frame: CudaFrame, start: Waypoint, goal: Waypoint, upper_bound: Waypoint = None, lower_bound: Waypoint = None):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.frame = frame
        self.start = start
        self.goal = goal
    
    def width(self) -> int:
        if isinstance(self.frame, CudaFrame):
            return self.frame.get_shape()[1]
        return self.frame.shape[1]

    def height(self) -> int:
        if isinstance(self.frame, CudaFrame):
            return self.frame.get_shape()[0]
        return self.frame.shape[0]
    
    def real_width(self) -> float:
        return self.width() * PROPORTION_meters_per_px
    
    def real_height(self) -> float:
        return self.height() * PROPORTION_meters_per_px


class TestFrame:
    __file: str
    __vehicle_dimensions: tuple[int, int]
    
    def __init__(self, file: str, vehicle_dimensions: tuple[int, int] = (18, 40)):
        self.__file = file
        self.__vehicle_dimensions = vehicle_dimensions
        
    def get_data_cuda(self) -> TestData:
        return self.__process_frame(True)
    
    def get_data_cpu(self) -> TestData:
        return self.__process_frame(False)
    
    
    def __is_empty(self, p):
        return p[0] == 255 and p[1] == 255 and p[2] == 255
    
    def __is_start(self, p):
        return p[0] == 0 and p[1] == 255 and p[2] == 0
        
    def __is_goal(self, p):
        return p[0] == 0 and p[1] == 0 and p[2] == 255
        
    def __process_frame(self, cuda_img: bool):
        frame = np.array(cv2.imread(self.__file))
        if frame is None:
            raise ValueError(f"Could not read image from {self.__file}")
        
        new_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
        start = None
        goal = None
        
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if (i == 428 and j == 265):
                    pass
                if self.__is_empty(frame[i, j]):
                    new_frame[i, j] = [1.0, 0, 0]
                elif self.__is_start(frame[i, j]):
                    start = Waypoint(x=j, z=i, heading=0.0)
                    new_frame[i, j] = [1.0, 0, 0]
                elif self.__is_goal(frame[i, j]):
                    goal = Waypoint(x=j, z=i, heading=0.0)
                    new_frame[i, j] = [1.0, 0, 0]
                else:
                    new_frame[i, j] = [0, 0, 0]
                    
        
        lower_bound = Waypoint(x=start.x - int(self.__vehicle_dimensions[0]/2), z=start.z + int(self.__vehicle_dimensions[1]/2))        
        upper_bound = Waypoint(x=start.x + int(self.__vehicle_dimensions[0]/2), z=start.z - int(self.__vehicle_dimensions[1]/2))
        
        if cuda_img:
            img_ptr = CudaFrame(frame = new_frame, 
                        min_dist_x= PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                        min_dist_z= PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                        lower_bound= lower_bound,
                        upper_bound= upper_bound)
        else:
             img_ptr = new_frame
        
        return TestData(frame=img_ptr, start=start, goal=goal, upper_bound=upper_bound, lower_bound=lower_bound)
        

class TestUtils:
    
    def pre_process_gpu(data: TestData, gpu_frame: CudaFrame, max_steering_deg: int, vehicle_length_m: float) -> np.ndarray:
        graph = CudaGraph(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=max_steering_deg,
            vehicle_length_m=vehicle_length_m,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=-1,
            lower_bound_z=-1,
            upper_bound_x=-1,
            upper_bound_z=-1
        )
        TestUtils.timed_exec(graph.compute_boundaries, gpu_frame)
        gpu_frame.invalidate_cpu_frame()
        return gpu_frame.get_frame()
    
    def log_graph(rrt: FastRRT, frame: CudaFrame, file: str) -> None:
        cimg = frame.get_color_frame()
        nodes = rrt.export_graph_nodes()
            
        width = cimg.shape[1]
        height = cimg.shape[0]
            
        for n in nodes:
            x = n[0]
            z = n[1]
            if (x < 0 or x >= width): continue
            if (z < 0 or z >= height): continue
            
            if int(n[2]) == GRAPH_TYPE_NODE:
                cimg[z, x, :] = [255, 255, 255]
            elif int(n[2]) == GRAPH_TYPE_PROCESSING:
                cimg[z, x, :] = [0, 255, 0]
            elif int(n[2]) == GRAPH_TYPE_TEMP:
                cimg[z, x, :] = [0, 0, 255]     
            
        cv2.imwrite(file, cimg)
    
    def output_path_result(frame: CudaFrame, path: np.ndarray, output: str) -> None:
        if path is None:
            return
        #f = frame.get_color_frame()
        frame.invalidate_cpu_frame()
        fo = frame.get_frame()
        f = np.zeros((fo.shape[1], fo.shape[0], 3), dtype=np.uint8)
        
        for i in range(fo.shape[0]):
            for j in range(fo.shape[1]):      
                if (fo[j, i, 0] == 1.0):
                    f[j, i, :] = [255, 255, 255]
                else:
                    f[j, i, :] = [0, 0, 0]
        
        for i in range(path.shape[0]):
            x = int(path[i,0])
            z = int(path[i,1])
            if x < 0 or x >= f.shape[1]: continue
            if z < 0 or z >= f.shape[0]: continue
            f[z, x, :] = [255, 0 , 0]
            if x-1 > 0:
                f[z, x-1, :] = [255, 0 , 0]
            if x+1 < fo.shape[1]:
                f[z, x+1, :] = [255, 0 , 0]            
        
        cv2.imwrite(output, f)
        
    def output_2path_result(frame: CudaFrame, path: np.ndarray, path2: np.ndarray, output: str) -> None:
        if path is None:
            return
        #f = frame.get_color_frame()
        frame.invalidate_cpu_frame()
        fo = frame.get_frame()
        f = np.zeros((fo.shape[1], fo.shape[0], 3), dtype=np.uint8)
        
        for i in range(fo.shape[0]):
            for j in range(fo.shape[1]):
                if (fo[i, j, 0] == 1.0):
                    f[j, i, :] = [255, 255, 255]
                else:
                    f[j, i, :] = [0, 0, 0]
        
        for i in range(path.shape[0]):
            x = int(path[i,0])
            z = int(path[i,1])
            if x < 0 or x >= f.shape[1]: continue
            if z < 0 or z >= f.shape[0]: continue
            f[z, x, :] = [255, 0 , 0]
            
        for i in range(path2.shape[0]):
            x = int(path2[i,0])
            z = int(path2[i,1])
            if x < 0 or x >= f.shape[1]: continue
            if z < 0 or z >= f.shape[0]: continue
            f[z, x, :] = [0, 0 , 255]            
        
        cv2.imwrite(output, f)
        
    
    def output_path_result_cpu(frame: np.ndarray, path: list, output: str) -> None:
        
        f = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if (frame[i, j, 0] == 1.0):
                    f[i, j, :] = [255, 255, 255]
                else:
                    f[i, j, :] = [0, 0, 0]
        
        if path is not None:
            for p in path:
                x = int(p[0])
                z = int(p[1])
                if x < 0 or x >= frame.shape[1]: continue
                if z < 0 or z >= frame.shape[0]: continue
                f[z, x, :] = [255, 0 , 0]
                if x-1 > 0:
                    f[z, x-1, :] = [255, 0 , 0]
                if x+1 < frame.shape[1]:
                    f[z, x+1, :] = [255, 0 , 0]
        
        cv2.imwrite(output, f)

    def output_2_path_result_cpu(frame: np.ndarray, path1: list, path2: list, output: str) -> None:
        
        f = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if (frame[i, j, 0] == 1.0):
                    f[i, j, :] = [255, 255, 255]
                else:
                    f[i, j, :] = [0, 0, 0]
        
        if path1 is not None:
            for p in path1:
                x = int(p[0])
                z = int(p[1])
                f[z, x, :] = [255, 0 , 0]

        if path2 is not None:
            for p in path2:
                x = int(p[0])
                z = int(p[1])
                f[z, x, :] = [0, 0, 255]                
        
        cv2.imwrite(output, f)

    
    def output_obstacle_graph(frame: CudaFrame, output: str) -> None:
        shape = frame.get_shape()
        frame.invalidate_cpu_frame()
        orig_frame = frame.get_frame()
        raw = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (orig_frame[i, j, 2] == 1.0):
                    raw[i, j] = [255, 255, 255]
                else:
                    raw[i, j] = [0, 0, 0]
        cv2.imwrite(output, raw)
        
    def timed_exec(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[{func.__name__}] {1000 * execution_time:.6f} ms")
        return result
    
    def timed_loop_exec(name: str, init_func, func, *args, **kwargs):
        start_time = time.time()
        loop_count = 0
        if init_func != None:
            init_func()
        while func(*args, **kwargs):
            loop_count += 1
            # if loop_count % 100 == 0:
            #     print(f"\t (partial exec: {loop_count}) mean time: {1000 * ((time.time() - start_time)/loop_count):.6f} ms/loop")
            pass
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[{name}] total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
    
    def timed_loop_exec_count(name: str, max_loop: int, init_func, func, *args, **kwargs):
        start_time = time.time()
        loop_count = 0
        if init_func != None:
            init_func()

        while func(*args, **kwargs) and loop_count < max_loop:
            loop_count += 1
            pass
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[{name}] total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")

