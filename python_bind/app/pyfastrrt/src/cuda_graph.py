import ctypes.util
import ctypes
import numpy as np
from pydriveless import SearchFrame, angle
import os


class CudaGraph:
    __ptr: ctypes.c_void_p
    __width: int
    __height: int

    def __init__(self,
                 width: int,
                 height: int,
                 perception_width_m: float,
                 perception_height_m: float,
                 max_steering_angle_deg: float,
                 vehicle_length_m: float,
                 min_dist_x: int,
                 min_dist_z: int,
                 lower_bound_x: int,
                 lower_bound_z: int,
                 upper_bound_x: int,
                 upper_bound_z: int,
                 path_costs: np.ndarray,
                 max_curvature: float = -1
                 ):

        CudaGraph.setup_cpp_lib()

        costs = np.ascontiguousarray(np.concatenate(
            ([path_costs.shape[0]], path_costs)), dtype=np.float32)

        self.__ptr = CudaGraph.lib.cudagraph_initialize(
            width,
            height,
            perception_width_m,
            perception_height_m,
            max_steering_angle_deg,
            vehicle_length_m,
            min_dist_x,
            min_dist_z,
            lower_bound_x,
            lower_bound_z,
            upper_bound_x,
            upper_bound_z,
            costs,
            max_curvature)

        self.__width = width
        self.__height = height

    def __del__(self) -> None:
        if hasattr(CudaGraph, "lib"):
            CudaGraph.lib.cudagraph_destroy(self.__ptr)

    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(CudaGraph, "lib"):
            return

        lib_path = os.path.join(os.path.dirname(
            __file__), "../cpp", "libfastrrt.so")
        lib_path_driveless = os.path.join(
            os.path.dirname(__file__), "../cpp", "libdriveless.so")
        CudaGraph.lib = ctypes.CDLL(lib_path)
        ctypes.CDLL(lib_path_driveless, mode=ctypes.RTLD_GLOBAL)

        CudaGraph.lib.cudagraph_initialize.restype = ctypes.c_void_p
        CudaGraph.lib.cudagraph_initialize.argtypes = [
            ctypes.c_int,  # width
            ctypes.c_int,  # height
            ctypes.c_float,  # perceptionWidthSize_m
            ctypes.c_float,  # perceptionHeightSize_m
            ctypes.c_float,  # maxSteeringAngle_rad
            ctypes.c_float,  # vehicleLength
            ctypes.c_int,  # minDistance_x
            ctypes.c_int,  # minDistance_z
            ctypes.c_int,  # lowerBound_x
            ctypes.c_int,  # lowerBound_z
            ctypes.c_int,  # upperBound_x
            ctypes.c_int,  # upperBound_z
            # segmentationClassCost
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
            ctypes.c_float # max_curvature
        ]

        CudaGraph.lib.cudagraph_destroy.restype = None
        CudaGraph.lib.cudagraph_destroy.argtypes = [
            ctypes.c_void_p
        ]

        CudaGraph.lib.compute_apf_repulsion.restype = None
        CudaGraph.lib.compute_apf_repulsion.argtypes = [
            ctypes.c_void_p,  # graph ptr
            ctypes.c_void_p,  # cuda search frame ptr
            ctypes.c_float,   # Kr
            ctypes.c_int      # radius
        ]

        CudaGraph.lib.compute_apf_attraction.restype = None
        CudaGraph.lib.compute_apf_attraction.argtypes = [
            ctypes.c_void_p,  # graph ptr
            ctypes.c_void_p,  # cuda search frame ptr
            ctypes.c_float,   # Ka
            ctypes.c_int,     # goal_x
            ctypes.c_int     # goal_z
        ]

        CudaGraph.lib.get_intrinsic_costs.restype = ctypes.POINTER(
            ctypes.c_float)
        CudaGraph.lib.get_intrinsic_costs.argtypes = [
            ctypes.c_void_p,  # graph ptr
        ]

        CudaGraph.lib.destroy_intrinsic_costs_ptr.restype = None
        CudaGraph.lib.destroy_intrinsic_costs_ptr.argtypes = [
            ctypes.c_void_p,  # costs ptr
        ]

        CudaGraph.lib.add.restype = None
        CudaGraph.lib.add.argtypes = [
            ctypes.c_void_p,  # graph ptr
            ctypes.c_int,     # x
            ctypes.c_int,     # z
            ctypes.c_float,   # heading
            ctypes.c_int,     # parent_x
            ctypes.c_int,     # parent_z
            ctypes.c_float    # cost
        ]
        
        CudaGraph.lib.add_temporary.restype = None
        CudaGraph.lib.add_temporary.argtypes = [
            ctypes.c_void_p,  # graph ptr
            ctypes.c_int,     # x
            ctypes.c_int,     # z
            ctypes.c_float,   # heading
            ctypes.c_int,     # parent_x
            ctypes.c_int,     # parent_z
            ctypes.c_float    # cost
        ]
        

        CudaGraph.lib.derivate_node.restype = None
        CudaGraph.lib.derivate_node.argtypes = [
            ctypes.c_void_p,   # graph ptr
            ctypes.c_void_p,   # frame ptr
            ctypes.c_float,    # steering_angle
            ctypes.c_int,      # path_size
            ctypes.c_float,    # velocity
            ctypes.c_int,      # x
            ctypes.c_int       # z
        ]

        CudaGraph.lib.accept_derived_nodes.restype = None
        CudaGraph.lib.accept_derived_nodes.argtypes = [
            ctypes.c_void_p,    # graph ptr
            ctypes.c_int,       # goal_x 
            ctypes.c_int,       # goal_z
            ctypes.c_float      # goal_heading
        ]

        CudaGraph.lib.check_in_graph.restype = ctypes.c_bool
        CudaGraph.lib.check_in_graph.argtypes = [
            ctypes.c_void_p,   # graph ptr
            ctypes.c_int,      # x
            ctypes.c_int       # z
        ]

        CudaGraph.lib.get_heading.restype = ctypes.c_float
        CudaGraph.lib.get_heading.argtypes = [
            ctypes.c_void_p,   # graph ptr
            ctypes.c_int,      # x
            ctypes.c_int       # z
        ]

        CudaGraph.lib.clear.restype = None
        CudaGraph.lib.clear.argtypes = [
            ctypes.c_void_p   # graph ptr
        ]

        CudaGraph.lib.list_all.restype = ctypes.POINTER(ctypes.c_float)
        CudaGraph.lib.list_all.argtypes = [
            ctypes.c_void_p   # graph ptr
        ]

        CudaGraph.lib.free_list_all.restype = None
        CudaGraph.lib.free_list_all.argtypes = [
            ctypes.POINTER(ctypes.c_float)
        ]

        CudaGraph.lib.solve_collisions.restype = None
        CudaGraph.lib.solve_collisions.argtypes = [
            ctypes.POINTER(ctypes.c_float)
        ]

        CudaGraph.lib.process_direct_goal_connection.restype = None
        CudaGraph.lib.process_direct_goal_connection.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float
        ]

        CudaGraph.lib.is_directly_connected_to_goal.restype = ctypes.c_bool
        CudaGraph.lib.is_directly_connected_to_goal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int
        ]

        CudaGraph.lib.direct_connection_to_goal_cost.restype = ctypes.c_float
        CudaGraph.lib.direct_connection_to_goal_cost.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int
        ]

        CudaGraph.lib.direct_connection_to_goal_heading.restype = ctypes.c_float
        CudaGraph.lib.direct_connection_to_goal_heading.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int
        ]

        CudaGraph.lib.dump_graph_to_file.restype = None
        CudaGraph.lib.dump_graph_to_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p
        ]
        
        CudaGraph.lib.read_from_dump_file.restype = None
        CudaGraph.lib.read_from_dump_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p
        ]

        CudaGraph.lib.get_parent.restype = ctypes.POINTER(ctypes.c_int)
        CudaGraph.lib.get_parent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,       # x
            ctypes.c_int,       # z
        ]
        
        CudaGraph.lib.free_parent_data.restype = None
        CudaGraph.lib.free_parent_data.argtypes = [
            ctypes.POINTER(ctypes.c_int)   # data_ptr
        ]
        
        CudaGraph.lib.get_cost.restype = ctypes.c_float
        CudaGraph.lib.get_cost.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,       # x
            ctypes.c_int,       # z
        ]
        
        CudaGraph.lib.count_all.restype = ctypes.c_int
        CudaGraph.lib.count_all.argtypes = [
            ctypes.c_void_p
        ]

    def get_parent(self, x: int, z: int) -> tuple[int, int]:
        ptr = CudaGraph.lib.get_parent(self.__ptr, x, z)
        res = (int(ptr[0]), int(ptr[1]))
        CudaGraph.lib.free_parent_data(ptr)
        return res

    def get_cost(self, x: int, z: int) -> tuple[int, int]:
        return CudaGraph.lib.get_cost(self.__ptr, x, z)



    def compute_apf_repulsion(self, cuda_ptr: SearchFrame, kr: float, radius: int):
        CudaGraph.lib.compute_apf_repulsion(
            self.__ptr, cuda_ptr.get_cuda_frame(), kr, radius)

    def compute_apf_attraction(self, cuda_ptr: SearchFrame, ka: float, goal_x: int, goal_z: int):
        CudaGraph.lib.compute_apf_attraction(
            self.__ptr, cuda_ptr.get_cuda_frame(), ka, goal_x, goal_z)

    def get_intrinsic_costs(self) -> np.ndarray:

        costs_ptr = CudaGraph.lib.get_intrinsic_costs(self.__ptr)

        res = np.zeros((self.__height, self.__width), dtype=np.float32)

        for h in range(self.__height):
            for w in range(self.__width):
                res[h, w] = float(costs_ptr[h * self.__width + w])

        CudaGraph.lib.destroy_intrinsic_costs_ptr(costs_ptr)
        return res

    def add(self, x: int, z: int, a: angle, parent_x: int, parent_z: int, cost: float) -> None:
        CudaGraph.lib.add(self.__ptr, x, z, a.rad(), parent_x, parent_z, cost)

    def add_temporary(self, x: int, z: int, a: angle, parent_x: int, parent_z: int, cost: float) -> None:
        CudaGraph.lib.add_temporary(self.__ptr, x, z, a.rad(), parent_x, parent_z, cost)


    def derivate_node(self, frame: SearchFrame, steering_angle: angle, path_size: int, velocity: float, parent_x: int, parent_z: int) -> None:
        CudaGraph.lib.derivate_node(self.__ptr, frame.get_cuda_ptr(
        ), steering_angle.rad(), path_size, velocity, parent_x, parent_z)

    def accept_derived_nodes(self) -> None:
        CudaGraph.lib.accept_derived_nodes(self.__ptr)

    def check_in_graph(self, x: int, z: int) -> bool:
        return CudaGraph.lib.check_in_graph(self.__ptr, x, z)

    def get_heading(self, x: int, z: int) -> angle:
        heading_rad = CudaGraph.lib.get_heading(self.__ptr, x, z)
        return angle.new_rad(heading_rad)

    def clear(self) -> None:
        CudaGraph.lib.clear(self.__ptr)


    def list_all(self) -> np.ndarray:
        ptr = CudaGraph.lib.list_all(self.__ptr)
        count = int(ptr[0])
        res = np.zeros((count, 4), dtype=np.float32)

        for i in range(0, count):
            pos = 4*i + 1
            res[i, 0] = float(ptr[pos])
            res[i, 1] = float(ptr[pos + 1])
            res[i, 2] = float(ptr[pos + 2])
            res[i, 3] = float(ptr[pos + 3])
        
        CudaGraph.lib.free_list_all(ptr)
        
        return res

    def solve_collisions(self) -> None:
        CudaGraph.lib.solve_collisions(self.__ptr)

    def process_direct_goal_connection (self, search_frame: SearchFrame, goal_x: int, goal_z: int, goal_heading: angle, max_curvature: float = -1) -> None:
        CudaGraph.lib.process_direct_goal_connection(self.__ptr, search_frame.get_cuda_ptr(), goal_x, goal_z, goal_heading.rad(), max_curvature)

    def is_directly_connected_to_goal(self, x: int, z: int) -> bool:
        return CudaGraph.lib.is_directly_connected_to_goal(self.__ptr, x, z)

    def direct_connection_to_goal_cost(self, x: int, z: int) -> float:
        return CudaGraph.lib.direct_connection_to_goal_cost(self.__ptr, x, z);  

    def direct_connection_to_goal_heading(self, x: int, z: int) -> angle:
        heading_rad = CudaGraph.lib.direct_connection_to_goal_heading(self.__ptr, x, z)
        return angle.new_rad(heading_rad)

    def dump_graph_to_file (self, file: str) -> None:
        CudaGraph.lib.dump_graph_to_file(self.__ptr, file.encode('utf-8'))
        
    def read_from_dump_file (self, file: str) -> None:
        CudaGraph.lib.read_from_dump_file(self.__ptr, file.encode('utf-8'))

    def get_parent(self, x: int, z: int) -> tuple[int, int]:
        ptr = CudaGraph.lib.get_parent(self.__ptr, x, z)
        res = (int(ptr[0]), int(ptr[1]))
        CudaGraph.lib.free_parent_data(ptr)
        return res

    def get_cost(self, x: int, z: int) -> tuple[int, int]:
        return CudaGraph.lib.get_cost(self.__ptr, x, z)

    def count_all(self) -> int:
        return CudaGraph.lib.count_all(self.__ptr)