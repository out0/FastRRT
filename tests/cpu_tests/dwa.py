import numpy as np
import math
import cv2

def euclidean_dist(p1: tuple[int, int, float], p2: tuple[int, int, float]) -> float:
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    return math.sqrt(dx ** 2 + dz ** 2)

def are_close(p1: tuple[int, int, float], p2: tuple[int, int, float]) -> bool:
    return  math.isclose(p1[0], p2[0], rel_tol=1e-2) and\
            math.isclose(p1[1], p2[1], rel_tol=1e-2)
            
def min_max_normalize(data):

    data = np.array(data)

    max_data = max(data)
    min_data = min(data)

    if min_data == -float('inf'):
        return 0

    if max_data - min_data == 0:
        data = [0.0 for i in range(len(data))]
    else:
        data = (data - min_data) / (max_data - min_data)

    return data

class EgoState:
    v: float
    w: float
    pos: tuple[int, int, float]
    
    def __init__(self, v: float, w: float, pos: tuple[int, int, float]):
        self.v = v
        self.w = w
        self.pos = pos
        pass

class GoalList:
    poses: list[tuple[int, int, float]]
    
    def __init__(self, lst: list[tuple[int, int, float]]):
        self.poses = lst
        pass
    
    def __dot(self, p1: tuple[int, int, float], p2: tuple[int, int, float]) -> float:
        return p1[0] * p2[0] + p1[1] * p2[1]
    
    def __project_on_path(self, p1: tuple[int, int, float], p2: tuple[int, int, float], p: tuple[int, int, float]) -> tuple[tuple[int, int, float], float, float]:
                
        path_size = euclidean_dist(p1, p2)
        
        if path_size == 0:
            return (None, 0, 0)
        
        l = (
            (p2[0] - p1[0]) / path_size,
            (p2[1] - p1[1]) / path_size,
            0, 0
        )
        v = (
            (p[0] - p1[0]),
            (p[1] - p1[1]),
            0, 0
        )
        distance_from_p1 = self.__dot(v, l)

        return (
            round(p1[0] + l[0] * distance_from_p1),
            round(p1[1] + l[1] * distance_from_p1),
            0, 0
        ), distance_from_p1, path_size
    
    def find_next(self, location: tuple[int, int, float], start: int) -> int:
        n = len(self.poses)
        
        if start == n - 1:
            _, distance_from_p1, path_size = self.__project_on_path(self.poses[start - 1], self.poses[start], location)
            if distance_from_p1 >= path_size:
                # the location is after the end of the line
                return -1
            else:
                return start
        
        return self.__find_nearest_next_pose_from_location(location, self.poses, start)

    def __find_mid (cls, p1: tuple[int, int, float], p2: tuple[int, int, float]) -> tuple[int, int]:
        return (p2[0] + p1[0])/2,  (p2[1] + p1[1])/2,

    def __find_nearest_next_pose_from_location(self, location:  tuple[int, int, float], poses: list[tuple[int, int, float]], start: int = 0, max_dist = 100):
        n = len(poses)
        best = -1
        best_dist = 999999999
        best_walked_path_ratio = 0
        best_in_middle = False
        
        for i in range (start, n - 1):
            
            if are_close(location, poses[i]):
                return i+1
            
            _, proportion, path_size = self.__project_on_path(poses[i], poses[i+1], location)
            
            if path_size == 0:
                continue
            
            if proportion >= path_size: # I'm after the path
                continue

            mx, my = self.__find_mid(poses[i], poses[i+1])
            
            dist = euclidean_dist(location, (mx, my, 0))
            
            if (dist > max_dist):
                continue
            
            if dist < best_dist:
                best_dist = dist
                best = i + 1
                best_in_middle = proportion > 0
                if best_in_middle:
                    best_walked_path_ratio = proportion / path_size
        
        if best_in_middle and best_walked_path_ratio >= 0.7:
            best = best + 1
        
        if best == n:
            #the best is the last point
            return -1
        
        return best


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
        
        cv2.imwrite(output, f)

class DWAInterpolation:
    """ This class implements the Dynamic Window Approach (DWA) algorithm to connect two points (P_start, P_goal), avoiding objects
        and using the Kinematic model to compute a set of paths. The best path, given a score is selected to predict the vehicle from
        P_ego --> P_goal
    """ 
    
    __frame: np.ndarray
    __D: float
    __window_size: int
    __dist_to_goal_tolerance: float
    __max_accel: float
    __max_w_accel: float
    __max_w: float
    __max_v: float
    __class_costs: list[float]
    
    __delta_v: float = 0.1
    __delta_w: float = 0.01
    __dt: float = 0.1
    __path_step: int = 20
    __state: EgoState
    
    #WEIGHT_H = 0.04
    WEIGHT_H = 1
    WEIGHT_V = 0.2
    WEIGHT_OBS = 0.1
    
    def __init__(self, 
                 frame: np.ndarray,
                 class_costs: list[float],
                 real_width_size_m: float,
                 real_height_size_m: float,
                 window_size: int,
                 dist_to_goal_tolerance: float):
        self.__frame = frame
        self.__class_costs = class_costs
        self.__max_accel = 0.5  # m/s^2
        self.__max_w_accel = 0.5 # rad/s²
        self.__max_w = 1 # rad/s
        self.__max_v = 1 # m/s
        self.__dt: float = 0.1
        self.__state = EgoState(0, 0, [0, 0, 0.0])
        self.__dist_to_goal_tolerance = dist_to_goal_tolerance
        self.__window_size = window_size

        real_size_m = math.sqrt(real_width_size_m ** 2 + real_height_size_m ** 2)
        px_size = math.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)
        
        r =  (px_size/real_size_m)
        self.__max_accel = r * self.__max_accel
        self.__max_v = r * self.__max_v

        self.__D = int(0.5*max(self.__frame.shape[0], self.__frame.shape[1]))
    
    
    def __draw_path_candidate_raw(self, pos: tuple[int, int, float], v_px: float, w: float, steps: int, dt: float):        
        p_start = pos
        
        path = []
        path.append(p_start)
        
        x, z, h = p_start
        h = h - math.pi/2
        last_x, last_z = p_start[0], p_start[1]
               
        
        for j in range(steps):
            x += v_px * dt * math.cos(h)
            z += v_px * dt * math.sin(h)
            h += w * dt
            
            if j >= 490:
                pass
            
            xc = int(x)
            zc = int(z)
            
            
            if (xc < 0 or xc >= self.__frame.shape[1]):
                break
            if (zc < 0 or zc >= self.__frame.shape[0]):
                break            
            
            if last_x == xc and last_z == zc: continue           
            
            path.append((xc, zc, h + math.pi/2))
            last_x = xc
            last_z = zc
        
        #output_path_result_cpu(self.__frame, path, "output1.png")     
        return path
    
    def __draw_v_proportional_path_candidate(self,
                        pos: tuple[int, int, float], 
                        v_px: float, 
                        w_rad_s: float, 
                        limit_dist: float):
        dt = 0.01
        steps_total = int(min((self.__D/(self.__max_v*dt)), limit_dist/dt))
        
        #print (f"steps_total = {steps_total}")
        return self.__draw_path_candidate_raw(pos, v_px, w_rad_s, steps=steps_total, dt=dt)


    
    def __draw_path_candidates(self, goal: tuple[int, int, float]) -> list[tuple[float, float, list[tuple[int, int, float]]]]:
        Rw = self.__dt * self.__max_w_accel
        min_w = self.__clip_w(self.__state.w- Rw)
        max_w = self.__clip_w(self.__state.w + Rw)
            
        Rv = self.__dt * self.__max_accel
        min_v = self.__clip_v(self.__state.v - Rv)
        max_v = self.__clip_v(self.__state.v + Rv)
        
        candidates = []
            
        limit_size = euclidean_dist(self.__state.pos, goal)

        # print (f"angles will vary from {min_w} to {max_w}")
        # print (f"speed will vary from {min_v} to {max_v}")
        for w in np.arange(min_w, max_w, self.__delta_w):
            for v in np.arange(min_v, max_v, self.__delta_v):
                if v == 0: continue
                #print ((v, w))
                path = self.__draw_v_proportional_path_candidate(self.__state.pos, v, w, limit_size)
                candidates.append((v, w, path))
        
        return candidates
            
    def __select_best_candidate(self, candidates: list[tuple[float, float, list[tuple[int, int, float]]]], goal: tuple[int, int, float]) -> tuple[float, float, list[tuple[int, int, float]]]:
        score_h = []
        score_v = []
        score_obs = []
            
        i = 0
        for c in candidates:
            v, w, path = c
            last_p = path[-1]
            score_h.append(self.__heading_angle_error_score(last_p, goal))
            score_v.append(v)
            score_obs.append(self.__collision_score(path))
            
        for s in [score_h, score_v, score_obs]:
            s = min_max_normalize(s)
            
        best_score = 0.0
        optimal_path_candidate = None
        

        for i in range(len(candidates)):
            s = DWAInterpolation.WEIGHT_H * score_h[i] + \
                DWAInterpolation.WEIGHT_V * score_v[i] + \
                DWAInterpolation.WEIGHT_OBS * score_obs[i]
                
            print (f"score = {s}")
                
            if s > best_score:
                best_score = s
                optimal_path_candidate = candidates[i]
        
        return optimal_path_candidate
        
    
    def interpolate(self, path: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        lst = GoalList(path)
        self.__state = EgoState(0, 0, path[0])
        
        i = 1
        res = []
        res.extend(path)
        
        i = 1
        while i > 0:
            print (f"navigating {i-1} -> {i}")
            new_state, new_path = self.navigate_towards(path[i])
            if new_state is None:
                i += 1
                continue
            self.__state = new_state
            if euclidean_dist(self.__state.pos, path[i]) <= 5:
                i += 1
        
            res.extend(new_path)
            output_path_result_cpu(self.__frame, res, 'output1.png')
        return res
    
    def interpolate2(self, path: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        self.__state = EgoState(0, 0, path[0])
        
        i = 1
        res = []
        res.extend(path)
        
        #best_dist_to_next = 99999999  
        while euclidean_dist(self.__state.pos, path[-1]) >= 5:
            print (f"navigating {i-1} -> {i}")
            if i >= len(path):
                break
            new_state, new_path = self.navigate_towards(path[i])
            if new_state is None:
                i += 1
                continue
            self.__state = new_state
            
            if new_state is None:
                i += 1
                #best_dist_to_next = float('inf')
                continue

            if euclidean_dist(self.__state.pos, path[i]) < 5:
                i += 1
            
            res.extend(new_path)
            output_path_result_cpu(self.__frame, res, 'output1.png')
            # if d < best_dist_to_next:
            #     best_dist_to_next = d
            #     res.extend(new_path)
            #     self.__state = new_state
            #     output_path_result_cpu(self.__frame, res, 'output1.png')
            # else:
            #     i += 1
            #     best_dist_to_next = float('inf')
            #     continue
        
        return res

    def navigate_towards(self, goal: tuple[int, int, float]) -> list[tuple[int, int, float]]:
        candidates = self.__draw_path_candidates(goal)
        
        # pp = []
        # for c in candidates:
        #     pp.extend(c[2])
        # output_path_result_cpu(self.__frame, pp, 'output1.png')
        
        best = self.__select_best_candidate(candidates, goal)
        if best is None:
            return None, None
        
        v, w, path = best
        return EgoState(v, w, path[-1]), path
        
        
    def __clip_w (self, w: float) -> float:
        if w < -self.__max_w:
            return -self.__max_w
        if w > self.__max_w:
            return self.__max_w
        return w
    
    def __clip_v (self, v: float) -> float:
        if v < 0:
            return 0
        if v > self.__max_v:
            return self.__max_v
        return v
    
    def __clip_angle(self, angle: float):
        if angle > math.pi:
            while angle > math.pi:
                angle -=  2 * math.pi
        elif angle < -math.pi:
            while angle < -math.pi:
                angle += 2 * math.pi
        return angle
    
    def __compute_path_heading(self, p1: tuple[int, int, float], p2: tuple[int, int, float]) ->float:
        dz = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        return math.pi/2 - math.atan2(-dz, dx)
        
    
    def __heading_angle_error_score(self, last_p: tuple[int, int, float], p_goal: tuple[int, int, float]) -> float:
        last_h = last_p[2]

        score = self.__compute_path_heading(last_p, p_goal) - last_h
        #print (f"path angle: {math.pi/2 - math.atan2(-dz, dx)}, last angle: {last_h}")
        score = math.pi - abs(self.__clip_angle(score))
        return score
    
    def __collision_score(self, path: list[tuple[int, int, float]]) -> float:
        score = 999999999
        
        obstacles = []
        x0, z0, _ = path[0]
        
        last_x = -1
        last_z = -1
        
        step = self.__window_size
        
        for i in range (-step, step):
            for j in range (-step, step):
                x = x0 + i
                z = z0 + j
                if x < 0 or x >= self.__frame.shape[1]: continue
                if z < 0 or z >= self.__frame.shape[0]: continue
                
                xc = int(x)
                zc = int(z)
                
                if xc == last_x and zc == last_z: continue
                last_x = xc
                last_z = zc
                
                segmentation_class = int(self.__frame[zc, xc, 0])
                if self.__class_costs[segmentation_class] < 0:
                    obstacles.append((x, z, 0.0))
            
        if len(obstacles) == 0:
            return 2.0
        
        for o in obstacles:
            for p in path:           
                d = euclidean_dist(p, o)
                if d < score:
                    score = d
                if self.__frame[p[1], p[0], 2] == 0.0:
                    return -float('inf')
        
        return score
    
    
    
    
    