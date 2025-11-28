import sys
import cv2, numpy as np, json
import math
from pydriveless import SearchFrame
from ensemble import PhysicalParameters
from pydriveless import Waypoint
from pyfastrrt import FastRRT, CudaGraph
import time
import re, os

PROPORTION_meters_per_px = 0.135316469
GRAPH_TYPE_NODE = 1
GRAPH_TYPE_TEMP = 2
GRAPH_TYPE_PROCESSING = 3

OG_REAL_WIDTH: float = 34.641016151377535
OG_REAL_HEIGHT: float = 34.641016151377535
MIN_DISTANCE_WIDTH_PX: int = 22
MIN_DISTANCE_HEIGHT_PX: int = 40
MAX_STEERING_ANGLE: int = 40
VEHICLE_LENGTH_M: float = 5.412658774  # num px * (OG_REAL_HEIGHT / OG_HEIGHT)
EGO_LOWER_BOUND: tuple[int, int] = (119, 148) 
EGO_UPPER_BOUND: tuple[int, int] =  (137, 108)
SEGMENTED_COLORS = np.array([
        [0,   0,   0],
        [128,  64, 128],
        [244,  35, 232],
        [70,  70,  70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170,  30],
        [220, 220,   0],
        [107, 142,  35],
        [152, 251, 152],
        [70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [0,   0, 142],
        [0,   0,  70],
        [0,  60, 100],
        [0,  80, 100],
        [0,   0, 230],
        [119,  11,  32],
        [110, 190, 160],
        [170, 120,  50],
        [55,  90,  80],     # other
        [45,  60, 150],
        [157, 234,  50],
        [81,   0,  81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180]
    ])

SEGMENTATION_CLASS_COST = np.array([
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        0,
        0,   # LAMP? investigate...
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1, # car
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        -1,
        0,
        0,
        0,
        0,
        -1
    ], dtype=np.float32)


class TestData:
    frame: SearchFrame
    cpu_frame: np.ndarray
    start: Waypoint
    goal: Waypoint
    upper_bound: Waypoint
    lower_bound: Waypoint
    
    def __init__(self, frame: SearchFrame, cpu_frame: np.ndarray, start: Waypoint, goal: Waypoint, upper_bound: Waypoint = None, lower_bound: Waypoint = None):
        self.cpu_frame = cpu_frame
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.frame = frame
        self.start = start
        self.goal = goal
    
    def width(self) -> int:
        if isinstance(self.frame, SearchFrame):
            return self.frame.width()
        return self.frame.shape[1]

    def height(self) -> int:
        if isinstance(self.frame, SearchFrame):
            return self.frame.height()
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
        
    def get_data_cuda(self, cost_map: bool = False, start: Waypoint = Waypoint(0,0,0), goal: Waypoint = Waypoint(0,0,0)) -> TestData:
        if cost_map:
            return self.__process_cost_map(True, start=start, goal=goal)
        return self.__process_frame(True)
    
    def get_data_cpu(self) -> TestData:
        return self.__process_frame(False)
    
    
    def __is_empty(self, p):
        return p[0] == 255 and p[1] == 255 and p[2] == 255
    
    def __is_start(self, p):
        return p[0] == 0 and p[1] == 255 and p[2] == 0
        
    def __is_goal(self, p):
        return p[0] == 0 and p[1] == 0 and p[2] == 255
    

    def read_pfm(file_path):
        with open(file_path, 'rb') as f:
            header = f.readline().decode('utf-8').rstrip()
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise ValueError("Not a PFM file.")

            dims_line = ''
            while True:
                line = f.readline().decode('utf-8')
                if line.startswith('#'):
                    continue  # skip comments
                dims_line = line
                break
            width, height = map(int, dims_line.strip().split())

            scale = float(f.readline().decode('utf-8').strip())
            endian = '<' if scale < 0 else '>'  # little endian if scale < 0
            scale = abs(scale)

            data = np.fromfile(f, endian + 'f')
            shape = (height, width, 3) if color else (height, width)

            data = np.reshape(data, shape)
            data = np.flipud(data)  # PFM stores pixels from bottom to top

            return data

    def __process_cost_map(self, cuda_img: bool, start: tuple[int, int, float], goal: tuple[int, int, float]):

        frame = TestFrame.read_pfm(self.__file)
        if frame is None:
            raise ValueError(f"Could not read image from {self.__file}")
        
        new_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
        
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if np.isfinite(frame[i, j]):  # unknown == Obstacle
                    new_frame[i, j] = [1.0, 255*float(frame[i, j])/0.75, 0]
                else:
                    new_frame[i, j] = [0, 0, 0]                    
        
        if start is not None:
            lower_bound = Waypoint(x=start[0] - int(self.__vehicle_dimensions[0]/2), z=start[1] + int(self.__vehicle_dimensions[1]/2))        
            upper_bound = Waypoint(x=start[0] + int(self.__vehicle_dimensions[0]/2), z=start[1] - int(self.__vehicle_dimensions[1]/2))
        else:
            lower_bound = None
            upper_bound = None

        if cuda_img:
            img_ptr = SearchFrame(
                width=frame.shape[1],
                height=frame.shape[0],
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
            img_ptr.set_frame_data(new_frame)
            # img_ptr = SearchFrame(frame = new_frame, 
            #             min_dist_x= PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
            #             min_dist_z= PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            #             lower_bound= lower_bound,
            #             upper_bound= upper_bound)
        else:
             img_ptr = new_frame
        
        return TestData(frame=img_ptr, cpu_frame=new_frame.copy(), start=start, goal=goal, upper_bound=upper_bound, lower_bound=lower_bound)
        

    def __process_frame(self, cuda_img: bool):

        frame = np.array(cv2.imread(self.__file))
        #frame = np.array(cv2.imread(self.__file, cv2.IMREAD_GRAYSCALE))
        if frame is None:
            raise ValueError(f"Could not read image from {self.__file}")
        
        new_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
        start = Waypoint(0, 0, 0)
        goal = Waypoint(0, 0, 0)

        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
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
                    
        
        lower_bound = (start.x - int(self.__vehicle_dimensions[0]/2), start.z + int(self.__vehicle_dimensions[1]/2))
        upper_bound = (start.x + int(self.__vehicle_dimensions[0]/2), start.z - int(self.__vehicle_dimensions[1]/2))
        
        if cuda_img:
            img_ptr = SearchFrame(
                width=frame.shape[1],
                height=frame.shape[0],
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
        else:
             img_ptr = new_frame
        
        return TestData(frame=img_ptr, cpu_frame=new_frame, start=start, goal=goal, upper_bound=upper_bound, lower_bound=lower_bound)
        

class TestUtils:
    
    def pre_process_gpu(data: TestData, gpu_frame: SearchFrame, max_steering_deg: int, vehicle_length_m: float, copy_intrinsic_costs_from_frame: bool = False) -> np.ndarray:
        graph = CudaGraph(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=max_steering_deg,
            vehicle_length_m=vehicle_length_m,
            min_dist_x=10,
            min_dist_z=10,
            lower_bound_x=data.lower_bound.x,
            lower_bound_z=data.lower_bound.z,
            upper_bound_x=data.upper_bound.x,
            upper_bound_z=data.upper_bound.z,
        )
        TestUtils.timed_exec(graph.compute_boundaries, gpu_frame, copy_intrinsic_costs_from_frame)
        gpu_frame.invalidate_cpu_frame()
        return gpu_frame.get_frame()
    
    def log_graph(rrt: FastRRT, frame: SearchFrame, file: str) -> None:
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
    
    def show_start(frame: np.ndarray, p: tuple[float, float, float]):
        TestUtils.show_point(frame, p, color=[0, 255, 0])

    def show_goal(frame: np.ndarray, p: tuple[float, float, float]):
        TestUtils.show_point(frame, p, color=[0, 0, 255])


    def show_point(frame: np.ndarray, p: tuple[float, float, float], color = [255, 255, 255]):
        x, z = map(int, p[:2])  # Convert to int for pixel access

        h, w = frame.shape[:2]

        for dx, dz in [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1)]:
            xi, zi = x + dx, z + dz
            if 0 <= zi < h and 0 <= xi < w:
                frame[zi, xi] = color

    def output_path_result(frame: SearchFrame, path: np.ndarray, output: str) -> None:
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
        

        TestUtils.show_start(f, path[0])
        TestUtils.show_goal(f, path[-1])



        cv2.imwrite(output, f)
        
    def output_2path_result(frame: SearchFrame, path: np.ndarray, path2: np.ndarray, output: str) -> None:
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
                f[z, x, :] = [255, 0 , 0]
        
        cv2.imwrite(output, f)
    
    def output_obstacle_graph(frame: SearchFrame, output: str) -> None:
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

#define GRAPH_TYPE_NULL 0
#define GRAPH_TYPE_NODE 1
#define GRAPH_TYPE_TEMP 2
#define GRAPH_TYPE_PROCESSING 3
#define GRAPH_TYPE_COLLISION 4

    def output_graph_nodes(frame: SearchFrame, graph: CudaGraph, file: str) -> None:
        nodes = graph.list_all()
        f = frame.get_color_frame()

        for i in range(nodes.shape[0]):
            node_type = int(nodes[i, 3])
            x = int(nodes[i, 0])
            z = int(nodes[i, 1])
           
            match node_type:
                case 1:
                    f[z, x] = [0, 255, 0]               
                case 2:
                    f[z, x] = [128, 0, 128]
                case 3:
                    f[z, x] = [128, 128, 128]
                case 4:
                    f[z, x] = [0, 0, 255]

        cv2.imwrite(file, f)






def import_cv2():
    # Safe check for OpenCV build flags
    ci_and_not_headless = getattr(cv2, "ci_build", False) and not getattr(cv2, "headless", False)

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_FONTDIR", None)