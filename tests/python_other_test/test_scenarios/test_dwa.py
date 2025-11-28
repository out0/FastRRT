import sys, os
sys.path.append("../../")
sys.path.append("../../../")
import numpy as np, math
import cv2

def draw_path_candidate_raw(pos: tuple[int, int, float], v_px: float, w: float, steps: int, dt: float, max_dim: tuple[int, int] = None):
        
        p_start = pos
        
        path = []
        path.append(p_start)
        
        x, z, h = p_start
        h = h
        last_x, last_z = p_start[0], p_start[1]
               
        
        for j in range(steps):
            x += v_px * dt * math.cos(h)
            z += v_px * dt * math.sin(h)
            h += w * dt
            
            if j >= 490:
                pass
            
            xc = int(x)
            zc = int(z)
            
            if max_dim is not None:
                if (xc < 0 or xc >= max_dim[0]):
                    break
                if (zc < 0 or zc >= max_dim[1]):
                    break            
            
            if last_x == xc and last_z == zc: continue           
            
            path.append((xc, zc, h))
            last_x = xc
            last_z = zc
        
        #output_path_result_cpu(self.__frame, path, "output1.png")     
        return path

def draw_v_proportional_path_candidate(frame: np.array, 
                        pos: tuple[int, int, float], 
                        v_px: float, 
                        w_rad_s: float, 
                        limit_goal: tuple[int, int, float]):
    dt = 0.01
    #max_path_size = euclidean_dist(pos, limit_goal)
    
    #max_range = min(0.5*frame.shape[1], max_path_size)
    max_range = 0.5*frame.shape[1]
    v_max = 100
    
    r = v_px / v_max
    
    steps_total = int(max_range/(v_max*dt))
    
    return draw_path_candidate_raw(pos, v_px, w_rad_s, steps=steps_total, dt=dt, max_dim=(frame.shape[1], frame.shape[0]))

def draw_point(frame, pos):
    for i in range(-1, 2):
        frame[pos[1] + i, pos[0]] = [0, 255, 0]
        frame[pos[1], pos[0] + i] = [0, 255, 0]

def euclidean_dist(p1: tuple[int, int, float], p2: tuple[int, int, float]):
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    return math.sqrt(dx ** 2 + dz ** 2)


def test_curves():
    frame_size = (1000, 1000, 3)
    frame_size_m = (100, 100)
    
    d = (frame_size[0]/frame_size_m[0], frame_size[1]/frame_size_m[1])
        
    f = np.full(frame_size, 255, dtype=np.uint8)
    draw_point(f, (500, 400))
    
    v = 100.0
    
    paths = []
    for w in np.arange(-1, 1, 0.1):
        path = draw_v_proportional_path_candidate(f, pos=(500, 500, math.radians(-45.0) - math.pi/2), v_px=v, w_rad_s=w, limit_goal=(500, 400, 0))
        paths.extend(path)

    f = np.full(frame_size, 255, dtype=np.uint8)
    for p in paths:
        if p[0] < 0 or p[0] >= frame_size[1]: continue
        if p[1] < 0 or p[1] >= frame_size[0]: continue
        f[p[1], p[0]] = [0, 0, 0]
    cv2.imwrite("test.png", f)

if __name__ == "__main__":
    test_curves()