import numpy as np
import random, time
import math
from node_grid import *
import sys
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
from model.waypoint import Waypoint
from dwa import DWAInterpolation

MIN_DIST_NONE = 0
MIN_DIST_CPU = 1
MIN_DIST_GPU = 2
   
class BiDirectionalRRT:    
    _width: int
    _height: int
    _perception_width_m: float
    _perception_height_m: float
    _max_steering_angle_deg: float
    _vehicle_length_m: float
    _timeout_ms: int
    _min_dist_x: int
    _min_dist_z: int
    _lower_bound_x: int
    _lower_bound_z: int
    _upper_bound_x: int
    _upper_bound_z: int
    _max_path_size_px: float
    _dist_to_goal_tolerance_px: float
    _start: Waypoint
    _goal: Waypoint
    _velocity_m_s: float
    _img: np.ndarray
    _nodes_start: NodeGrid
    _nodes_goal: NodeGrid
    _goal_reached: bool
    _proc_start: float
    _class_cost: list[float]
    _goal_node_reached: int2
    _check_min_distances_cpu: bool
    _check_min_distances_gpu: bool

    
    def __init__(self, 
              width: int, 
              height: int,
              perception_width_m: float,
              perception_height_m: float,
              max_steering_angle_deg : float,
              vehicle_length_m: float,
              timeout_ms: int,
              min_dist_x: int,
              min_dist_z: int,
              lower_bound_x: int,
              lower_bound_z: int,
              upper_bound_x: int,
              upper_bound_z: int,
              max_path_size_px: float,
              dist_to_goal_tolerance_px: float,
              class_cost: list[float]
              ):
        self._width = width
        self._height = height
        self._perception_width_m = perception_width_m
        self._perception_height_m = perception_height_m
        self._max_steering_angle_deg = max_steering_angle_deg
        self._vehicle_length_m = vehicle_length_m
        self._timeout_ms = timeout_ms
        self._min_dist_x = int(min_dist_x/2)
        self._min_dist_z = int(min_dist_z/2)
        self._lower_bound_x = lower_bound_x
        self._lower_bound_z = lower_bound_z
        self._upper_bound_x = upper_bound_x
        self._upper_bound_z = upper_bound_z
        self._max_path_size_px = max_path_size_px
        self._dist_to_goal_tolerance_px = dist_to_goal_tolerance_px
        self._proc_start = 0
        self._class_cost = class_cost
        self._goal_reached = False
        self._check_min_distances_cpu = True
        self._check_min_distances_gpu = False

        
        
    def set_plan_data(self, img: np.ndarray, start: Waypoint, goal: Waypoint, velocity_m_s: float) -> bool:
        self._img = img
        self._start = (start.x, start.z, start.heading)
        self._goal = (goal.x, goal.z, goal.heading)
        self._velocity_m_s = velocity_m_s

    def check_dist_gpu(self) -> None:
        self._check_min_distances_cpu = False
        self._check_min_distances_gpu = True
   
    def list_nodes(self) -> list[int2]:
        if self._nodes_start == None:
            return[]
        return self._nodes_start.get_node_list()

    def list_goal_nodes(self) -> list[int2]:
        if self._nodes_goal == None:
            return[]
        return self._nodes_goal.get_node_list()
   


    def __initialize_node_grid(self, start_node: tuple[int, int, float], cpu_min_dist: int = MIN_DIST_NONE) -> None:
        grid = NodeGrid(self._width, self._height, (start_node[0], start_node[1]))
        grid.set_search_params((self._min_dist_x, self._min_dist_z),
                                      (self._lower_bound_x, self._lower_bound_z),
                                      (self._upper_bound_x, self._upper_bound_z))
        
        grid.set_physical_params(PhysicalParameters(
                (self._width / self._perception_width_m, self._height / self._perception_height_m),
                self._vehicle_length_m/2,
                math.radians(self._max_steering_angle_deg)))
        grid.set_class_costs(self._class_cost)
        
        if cpu_min_dist == MIN_DIST_CPU:
            grid.check_min_dist()
        elif cpu_min_dist == MIN_DIST_GPU:
            grid.check_min_dist_gpu()
            
        grid.set_heading(start_node[0], start_node[1], start_node[2])
        return grid


    def search_init(self, cpu_min_dist: int = MIN_DIST_NONE) -> None:
        self._proc_start = time.time()
        self._nodes_start = self.__initialize_node_grid(self._start, cpu_min_dist)
        self._nodes_goal = self.__initialize_node_grid(self._goal, cpu_min_dist)
        self._nodes_goal.set_heading(self._goal[0], self._goal[1], self._goal[2] + math.pi)
            
    
    def __sample(self) -> int2:
        return (
            random.randint(1, self._width - 1),
            random.randint(1, self._height - 1)
        )
    
    
    def __dist(self, p1: int2, p2: int2) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.hypot(dx, dy)
    
    def __steer(self, parent: int2, node: int2, max_step_size: float) -> int2:
        dx = node[0] - parent[0]
        dy = node[1] - parent[1]
        dist = np.hypot(dx, dy)
        if dist <= max_step_size:
            return (node[0], node[1])
        theta = np.arctan2(dy, dx)
        new_x = int(parent[0] + max_step_size * np.cos(theta))
        new_y = int(parent[1] + max_step_size * np.sin(theta))
        return (new_x, new_y)
    
    def __check_pos_limits(self, node: int2) -> bool:
        xc = node[0]
        zc = node[1]
        return xc >= 0 and xc < self._width and zc >= 0 and zc < self._height
    
    def __connect_new_node(self, grid: NodeGrid, parent: int2, node: int2, cost: float) -> bool:
        if not self.__check_pos_limits(node):
            return False
        
        grid.add_node(node, parent, cost)
        if (node[0] == parent[0] and node[1] == parent[1]):
            print("cyclic")
        
        return True
    
    def __check_timeout(self) -> bool:
        if self._timeout_ms <= 0:
            return False
        elapsed_time = (time.time() - self._proc_start) * 1000
        return elapsed_time > self._timeout_ms
    
    
    def __check_goal_reached(self, x_new_start: int2, kinematic: bool) -> bool:
        if x_new_start is None:
            return False
        
        x_min = None
        c_min = 999999999
        heading = -1
       
        X_near_lst = self._nodes_goal.find_near_nodes(x_new_start, self._max_path_size_px)
               
        for x_near in X_near_lst:
            dist = self.__dist(x_new_start, x_near)
            res = self.__check_connection(self._nodes_goal, x_near, x_new_start, dist, kinematic)
     
            if res.feasible and res.cost < c_min:
                x_min = x_near
                c_min = res.cost
                heading = res.final_heading
        
        if x_min is None:
            return False
        
        # connect both graphs
        n: int2 = x_min
        p: int2 = x_new_start
        h: float = heading
        last_n: int2 = None

        while (n[0] >= 0 and n[1] >= 0):
            self._nodes_start.add_node(node=n, parent=p, cost=self._nodes_start.get_cost(p[0], p[1]) + self._nodes_start.get_cost(n[0], n[1]))
            self._nodes_start.set_heading(n[0], n[1], h)
            p = n
            last_n = n
            h = self._nodes_goal.get_heading(n[0], n[1]) - math.pi
            if h < 0:
                h = h + 2* math.pi
            n = self._nodes_goal.get_parent(n)

            

        self._goal_reached = True
        self._goal_node_reached = last_n
        return True

    def timeout(self) -> bool:
        return self.__check_timeout()

    def __path_heading(self, p1, p2) -> float:
        dz = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        
        if dx == 0 and dz == 0: return 0
        return math.pi/2 - math.atan2(-dz, dx)


    def __check_connection(self, grid: NodeGrid, x_nearest: int2, x_new: int2, dist: float, kinematic: bool) -> GraphConnectionResult:
        if kinematic:
            return grid.check_kinematic_connection(self._img, x_nearest, x_new, self._velocity_m_s)
        else:
            heading = self.__path_heading(x_nearest, x_new)
            return GraphConnectionResult(
                heading, grid.check_direct_connection(self._img, x_nearest, x_new), grid.get_cost(x_nearest[0], x_nearest[1]) + dist)            
    

    def __equals(self, p1: int2, p2: int2) -> bool:
        return p1[0] == p2[0] and p1[1] == p2[1]
    
    def __loop_rrt_star_graph(self, grid: NodeGrid, x_rand: int2, kinematic: bool) -> None:
        
        x_nearest, dist = grid.find_nearest(x_rand)
        x_new = self.__steer(x_nearest, x_rand, self._max_path_size_px)
        if (x_nearest[0] == x_new[0] and x_nearest[1] == x_new[1]):
            return None
    
        x_min = None
        c_min = 999999999
        heading = -1
       
        X_near_lst = grid.find_near_nodes(x_new, self._max_path_size_px)
               
        for x_near in X_near_lst:
            res = self.__check_connection(grid, x_near, x_new, dist, kinematic)
     
            if res.feasible and res.cost < c_min:
                x_min = x_near
                c_min = res.cost
                heading = res.final_heading
        
        if x_min is None:
            return None

        x_new_parent = x_min
        if not self.__connect_new_node(grid, parent=x_new_parent, node=x_new, cost=c_min):
            #self.__test_cyclic_ref(x_new)
            return None
        
        grid.set_heading(x_new[0], x_new[1], heading)

        for x_near in X_near_lst:
            if self.__equals(x_near, x_new) or self.__equals(x_near, x_new_parent):
                continue
            
            res = self.__check_connection(grid, x_new, x_near, dist, kinematic)
            
            if res.feasible and res.cost < grid.get_cost(x_near[0], x_near[1]):
                #rewire
                grid.set_parent(x_near, new_parent=x_new)
                grid.set_heading(x_near[0], x_near[1], heading_rad=res.final_heading)

        return x_new

    def loop_rrt_star(self, kinematic: bool) -> bool: 
        x_rand: int2 = self.__sample() 
        x_new_start = self.__loop_rrt_star_graph(self._nodes_start, x_rand, kinematic)
        x_new_goal = self.__loop_rrt_star_graph(self._nodes_goal, x_rand, kinematic)
        self.__check_goal_reached(x_new_start, x_new_goal)
        return not self.__check_timeout()
       
 
    def goal_reached(self) -> bool:
        return self._goal_reached
    
    
    def __hermite_interpolation(self, p1: Waypoint, p2: Waypoint) -> list[tuple[int, int, float]]:     
        curve =[]
        d = Waypoint.distance_between(p1, p2)
        numPoints = int(round(d))
        if numPoints <= 2:
            return [(p1.x, p1.z), (p2.x, p2.z)] 
        a1 = p1.heading - (math.pi / 2)
        a2 = p2.heading - (math.pi / 2)
        
        # Tangent vectors
        tan1 = (d * math.cos(a1), d * math.sin(a1))
        tan2 = (d * math.cos(a2), d * math.sin(a2))

        last_x = -1
        last_z = -1
        
        for i in range (numPoints):
            t = float(i) / (numPoints - 1)

            t2 = t * t
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            x = h00 * p1.x + h10 * tan1[0] + h01 * p2.x + h11 * tan2[0]
            z = h00 * p1.z + h10 * tan1[1] + h01 * p2.z + h11 * tan2[1]

            if (x < 0 or x >= self._width):
                continue
            if (z < 0 or z >= self._height):
                continue

            cx = int(round(x))
            cz = int(round(z))

            if (cx == last_x and cz == last_z):
                continue

            if cx < 0 or cx > self._width: continue
            if cz < 0 or cz > self._height: continue
            if cx == last_x and cz == last_z: continue

            t00 =  6 * t2 - 6 * t
            t10 =  3 * t2 - 4 * t + 1
            t01 = -6 * t2 + 6 * t
            t11 =  3 * t2 - 2 * t

            ddx = t00 * p1.x + t10 * tan1[0] + t01 * p2.x + t11 * tan2[0]
            ddz = t00 * p1.z + t10 * tan1[1] + t01 * p2.z + t11 * tan2[1]

            heading = math.atan2(ddz, ddx) + 0.5*math.pi

            curve.append((cx, cz, heading))
            last_x = cx
            last_z = cz
        
        return curve

    
    def __interpolate_straight_line(self, p1: tuple[int, int, float], p2: tuple[int, int, float]) -> list[int2]:
        forward_movement = p1[1] > p2[1]
        
        dx = p2[0]- p1[0]
        dz = p2[1] - p1[1]
        
        if dz == 0:
            return []
        
        slope = dx / dz
        height = self._img.shape[0]

        dz = (2 * dz) / height

        z = p1[1]

        result: list[int2] = []

        if forward_movement:
            while z > p2[1]:
                x = p1[0] + (z - p1[1]) * slope
                z += dz
                result.append((math.floor(x), math.floor(z)))
        else:
            while z < p2[1]:
                x = p1[0] + (z - p1[1]) * slope
                z += dz
                result.append((math.floor(x), math.floor(z)))
        
        return result
    
    
    def __in_ego_boundaries(self, x: int, z: int) -> bool:
        return (
            self._lower_bound_x <= x <= self._upper_bound_x and
            self._upper_bound_z <= z <= self._lower_bound_z)
    
    def __check_feasible(self, frame: np.ndarray, p: int2) -> bool:
        
        if self._check_min_distances_cpu:      
            for z in range(p[1] - self._min_dist_z, p[1] + self._min_dist_z + 1):
                for x in range(p[0] - self._min_dist_x, p[0] + self._min_dist_x + 1):
                    if not self.__check_pos_limits((x, z)):
                        continue
                    
                    if self.__in_ego_boundaries(x, z):
                        continue
                    
                    segmentation_class = int(frame[z, x, 0])
                    if self._class_cost[segmentation_class] < 0:
                        return False
            return True
        elif self._check_min_distances_gpu:
            x, z = p
            return frame[z, x, 2] == 0
        else: # no checking
            return True
    
    def __simplify(self, path: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        i = 0
        n = len(path)
        
        new_path = []
        new_path.append(path[i])
        
        while i < n:
            j = i + 1
            
            if j >= n - 1:
                break
            
            fesible = True
            while fesible and j < n - 1:
                direct_path = self.__interpolate_straight_line(path[i], path[j + 1])
                for p in direct_path:
                    fesible = self.__check_feasible(self._img, p)
                    if not fesible:
                        break
                if fesible:
                    j += 1
            
            
            new_path.append(path[j])
            i = j

        new_path.append(path[-1])
        return new_path
    
    def __mid_point(self, p1: tuple[int, int, float], p2: tuple[int, int, float]) -> tuple[int, int, float]:
        x = (p1[0] + p2[0]) / 2
        z = (p1[1] + p2[1]) / 2
        heading = (p1[2] + p2[2]) / 2
        return (int(x), int(z), heading)
    
    def __brake_path(self, path: list[tuple[int, int, float]], window_size: float) -> tuple[list[tuple[int, int, float]], bool]:
        last = len(path) - 1
        
        new_path = []
        new_path.append(path[0])
        brake = False
            
        i = 0
        while i < last:
            j = i + 1
            if euclidean_distance(path[i], path[j]) > window_size:
                new_p = self.__mid_point(path[i], path[j])
                new_path.append(new_p)
                brake = True
            new_path.append(path[j])
            i = j
            
        return new_path, brake
            
            
                
                
    
    def optimize(self, fuse_dwa: bool) -> list[tuple[int, int, float]]:
        path = self.get_planned_path(False)
        path = self.__simplify(path)
        #np.savetxt("temp.dat", path, fmt="%f")
        #return 
        
        # return self.hermite_interpolate(path)

        # path = np.loadtxt("temp.dat")
        # if path.ndim == 1:
        #     path = path.reshape(1, -1)

        if fuse_dwa:
            # brake = True
            # while brake:
            #     path, brake = self.__brake_path(path, 30)
                
            dwa = DWAInterpolation(
                frame=self._img,
                class_costs=self._class_cost,
                real_height_size_m=self._perception_height_m,
                real_width_size_m=self._perception_width_m,
                window_size=30,
                dist_to_goal_tolerance=15.0
                )
            
            path = dwa.interpolate(path)
        
        return path
    
    def hermite_interpolate(self, path: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        intepolate_path = []
        p1 = None
        p2 = None
        for p in path:
            if p1 is None:
                p1 = Waypoint(p[0], p[1], p[2])
                continue
            p2 = Waypoint(p[0], p[1], p[2])
            partial_path = self.__hermite_interpolation(p1, p2)
            intepolate_path.extend(partial_path)
            p1 = p2
        return intepolate_path
    
    def get_planned_path(self, interpolate: bool = False):
        nearest_parent, dist = self._nodes_start.find_nearest((self._goal[0], self._goal[1]))
        
        path = []
        heading = self._nodes_start.get_heading(nearest_parent[0], nearest_parent[1])
        node = (int(nearest_parent[0]), int(nearest_parent[1]), heading)
        while node[0] != -1 and node[1] != -1:
            path.append(node)
            n = self._nodes_start.get_parent((node[0], node[1]))
            heading = self._nodes_start.get_heading(n[0], n[1])
            node = (n[0], n[1], heading)
        
        path.reverse()
        
        if not interpolate or len(path) < 2:
            return path

        return self.hermite_interpolate(path)
    
    # def get_planned_path(self, interpolate: bool = False):
    #     nearest_parent, dist = self._nodes.find_nearest((self._goal[0], self._goal[1]))
        
    #     path = []
    #     node = (int(nearest_parent[0]), int(nearest_parent[1]))
    #     while node[0] != -1 and node[1] != -1:
    #         path.append(node)
    #         node = self._nodes.get_parent(node)
        
        
    #     if interpolate and len(path) > 1:
    #         x = [p[0] for p in path]
    #         y = [p[1] for p in path]
            
    #         # Generate a parameter t for cubic spline interpolation
    #         t = np.arange(len(path))
            
    #         # Create cubic splines for x and y
    #         cs_x = CubicSpline(t, x)
    #         cs_y = CubicSpline(t, y)
            
    #         # Interpolate with finer granularity
    #         t_fine = np.linspace(0, len(path) - 1, num=10 * len(path))
    #         x_fine = cs_x(t_fine)
    #         y_fine = cs_y(t_fine)
            
    #         # Combine interpolated points into a new path
    #         path = [(int(xf), int(yf)) for xf, yf in zip(x_fine, y_fine)]
        
    #     path.reverse()
    #     return path