import math
import numpy as np
import matplotlib.pyplot as plt
import cv2, json


class CurveData:
    name: str
    curve: np.ndarray
    num_points: int
    jerk: float
    total_length: float
    tan_discontinuity: bool
    coarse: bool
    proc_time_ms: float
    timeout: bool
    goal_reached: bool
    
    def __init__(self, 
                 name: str,
                 coarse: bool,
                 curve: np.ndarray,
                 num_points: int,
                 jerk: float,
                 total_length: float,
                 tan_discontinuity: bool,
                 proc_time_ms: float,
                 num_loops: int,
                 timeout: bool,
                 goal_reached: bool):
        self.name = name
        self.curve = curve
        self.num_points = num_points
        self.jerk = jerk
        self.total_length = total_length
        self.tan_discontinuity = tan_discontinuity
        self.coarse = coarse
        self.proc_time_ms = proc_time_ms
        self.timeout = timeout
        self.num_loops = num_loops
        self.goal_reached = goal_reached

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "curve" : [],
            "type": "coarse" if self.coarse else "optim",
            "num_points": self.num_points,
            "jerk": self.jerk,
            "total_length": self.total_length,
            "tan_discontinuity": self.tan_discontinuity,
            "proc_time_ms": self.proc_time_ms,
            "timeout": self.timeout,
            "num_loops": self.num_loops,
            "goal_reached": self.goal_reached
            
        }
        for i in range(self.curve.shape[0]):
            data["curve"].append(f"({int(self.curve[i, 0])}, {int(self.curve[i, 1])}, {self.curve[i, 2]:0.2f})")
            
        return f"{json.dumps(data)}\n"
    
    def to_csv_header(self) -> str:
        return "\"name\";\"num_points\"; \"jerk\"; \"total_length\"; \"tan_discontinuity\"; \"coarse\"; \"proc_time_ms\"; \"timeout\"; \"num_loops\";\"goal_reached\"\n"
    
    def to_csv(self) -> str:
        disc = "yes" if self.tan_discontinuity else "no"
        curve_type = "coarse" if self.coarse else "optim"
        timeout = "yes" if self.timeout else "no"
        goal_reached = "yes" if self.goal_reached else "no"
        return f"\"{self.name}\";\"{self.num_points}\"; \"{self.jerk}\"; \"{self.total_length}\"; \"{disc}\"; \"{curve_type}\"; \"{self.proc_time_ms}\" \"{timeout}\"; \"{self.num_loops}\";\"{goal_reached}\"\n"

class CurveAssessment:
    __width: int
    __height: int
    __debug: bool
    
    def __init__(self, width: int, height: int, debug: bool = False):
        self.__width = width
        self.__height = height
        self.__debug = debug

    def __dist(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        return math.sqrt(dx ** 2 + dz ** 2)
    

    def __compute_curve_length(curve: np.ndarray) -> float:
        len: float = 0
        p1 = (curve[0, 0], curve[0, 1])
        for i in range(1, curve.shape[0]):
            p2 = (curve[i, 0], curve[i, 1])
            len += CurveAssessment.__dist(p1, p2)
            p1 = p2
        return len

    
    def assess_curve(self, curve: np.ndarray, start_heading: float, compute_heading: bool = True) -> CurveData:
        return CurveData(
            name=None,
            coarse=False,
            goal_reached=False,
            timeout=False,
            proc_time_ms=0.0,
            num_loops=0,
            curve=curve,
            num_points=curve.shape[0],
            total_length=CurveAssessment.__compute_curve_length(curve),
            jerk=self.__compute_jerk(curve),
            tan_discontinuity=self.__tangential_discontinuity(curve, window_side=4, threshold=15, start_heading=start_heading, compute_heading=compute_heading),
        )


    def draw_hermite(self, p1: tuple[int, int], h1: float, p2: tuple[int, int], h2: float) -> list[tuple[int, int]]:
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        d = math.sqrt(dx * dx + dz * dz)
        num_points = abs(dz)

        a1 = math.radians(h1) - 0.5*math.pi
        a2 = math.radians(h2) - 0.5*math.pi

        tan1 = (d * math.cos(a1), d * math.sin(a1))
        tan2 = (d * math.cos(a2), d * math.sin(a2))

        lastx = -1
        lastz = -1

        res = []

        for i in range(num_points):
            t = i / (num_points - 1)
            t2 = t ** 2
            t3 = t ** 3

            h00 = 2 * t3 - 3 * t2 + 1
            h10 =     t3 - 2 * t2 + t
            h01 =-2 * t3 + 3 * t2
            h11 =     t3     - t2     

            t00 =  6 * t2 - 6 * t
            t10 =  3 * t2 - 4 * t + 1
            t01 = -6 * t2 + 6 * t
            t11 =  3 * t2 - 2 * t

            x = h00 * p1[0] + h10 * tan1[0] + h01 * p2[0] + h11 * tan2[0]
            z = h00 * p1[1] + h10 * tan1[1] + h01 * p2[1] + h11 * tan2[1]

            if x < 0 or x > self.__width: continue
            if z < 0 or z > self.__height: continue
            if x == lastx and z == lastz: continue

            ddx = t00 * p1[0] + t10 * tan1[0] + t01 * p2[0] + t11 * tan2[0]
            ddz = t00 * p1[1] + t10 * tan1[1] + t01 * p2[1] + t11 * tan2[1]

            heading = math.atan2(ddz, ddx) + 0.5*math.pi

            res.append((int(round(x)), int(round(z)), heading))
            
            lastx = x
            lastz = z

        return res
    

    def __curve_heading(self, p1: tuple[int, int], p2: tuple[int, int]) -> tuple[float, bool]:
        if p1[0] == p2[0] and p1[1] == p2[1]: return 0.0, False

        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]

        h = (math.pi / 2) - math.atan2(-dz, dx)
        if h > math.pi:
            h = h - 2 * math.pi

        return h, True

    def __curve_mean_heading(self, curve: np.ndarray, pos: int, window_side: int = 2, debug = False):
        i = pos - window_side
        j = pos + window_side

        if i < 0:
            j = j - i
            i = 0
        if j >= curve.shape[0]:
            i = i - (j - curve.shape[0]) - 1
            j = curve.shape[0] - 1
        

        h = 0; count = 0

        for k in range(i, pos):
            p, valid = self.__curve_heading((curve[k,0], curve[k,1]), (curve[pos, 0], curve[pos, 1]))
            if not valid and debug:
                print (f"not valid for {k}, {pos}")
                continue
            h += p
            count += 1
            if debug:
                print (f"h({k}, {pos}) = {math.degrees(p)} current mean: {h/count}")
            
        for k in range(pos+1, j+1):
            p, valid = self.__curve_heading((curve[pos, 0], curve[pos, 1]), (curve[k,0], curve[k,1]))
            if not valid and debug:
                print (f"not valid for {pos}, {k}")
                continue
            h += p
            count += 1
            if debug:
                print (f"h({pos}, {k}) = {math.degrees(p)} current mean: {h/count}")

        return h/count

    def __tangential_discontinuity(self, curve: np.ndarray, window_side: int=2, threshold: float = 15, start_heading: float = 0.0, compute_heading: bool = True):
        #a_before = curve_mean_heading(curve, 0, window_side=window_side)
        a_before = start_heading
        threshold_rad = math.radians(threshold)
        for i in range(1, curve.shape[0]):
            if compute_heading:
                a = self.__curve_mean_heading(curve, i, window_side=window_side)
            else:
                a = curve[i, 2]
            if abs(a - a_before) > threshold_rad:
                return True
                #print(f"pos: {i} {(curve[i][0], curve[i][1])} spike: {a_before} -> {a}")
            a_before = a
        return False


    def __derivate_p(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        return math.atan2(dz,dx)

    def __derivate_curve_p(self, curve: list[int, int], pos: int, window_side: int = 2, debug = False):
        i = pos - window_side
        j = pos + window_side

        if i < 0:
            j = j - i
            i = 0
        if j >= len(curve):
            i = i - (j - len(curve)) - 1
            j = len(curve) - 1
        

        h = 0; count = 0

        for k in range(i, pos):
            p = self.__derivate_p(curve[k], curve[pos])
            h += p
            count += 1
            if self.__debug:
                print (f"h({k}, {pos}) = {p} current mean: {h/count}")
            
        for k in range(pos+1, j+1):
            p = self.__derivate_p(curve[pos], curve[k])        
            h += p
            count += 1
            if self.__debug:
                print (f"h({pos}, {k}) = {p} current mean: {h/count}")

        return h/count

    def __gradient(self, curve: list[int, int], window_side: int = 2, debug = False):
        dev_curve = []
        for i in range (0, len(curve)):
            dev_curve.append(self.__derivate_curve_p(curve, pos=i, window_side=window_side, debug=debug))
        return dev_curve

    def __sum(self, vals: list) -> float:
        v = 0
        for i in range (len(vals)):
            v += abs(vals[i])
        return v

    def __compute_jerk(self, curve):
        v = self.__gradient(curve, window_side=4)
        a = np.gradient(v, 1)
        j = np.gradient(a, 1)
        return abs(self.__sum(j))
    

