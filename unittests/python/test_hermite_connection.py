import unittest, math, numpy as np
from pydriveless import angle
from pydriveless import MapPose
from pydriveless import SearchFrame
from pyfastrrt import FastRRT
from hermite import HermiteCurve
from test_utils import *
import cv2
import_cv2()

class TestHermiteConnenction(unittest.TestCase):

    def show_curve(self, frame: SearchFrame, path: np.ndarray):

        f = frame.get_frame()

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if int(f[i, j, 2]) & 256 > 0:
                    f[i, j, :] = [255, 255, 255]
                else:
                    f[i, j, :] = [0, 0, 0]

        for p in path:
            x, z = p.x, p.z
            if x < 0 or x >= frame.width() or z < 0 or z >=  frame.height():
                continue    
            f[z, x, :] = [0, 0, 255]

        cv2.imwrite("output1.png", f)

    def intepolate(self, start: Waypoint, goal: Waypoint, show_path: bool = False) -> bool:
        img = np.array(cv2.imread("long_map.png"), dtype=np.float32)

        frame = SearchFrame (
            width=img.shape[1],
            height=img.shape[0],
            lower_bound=(-1, -1),
            upper_bound=(-1, -1),
        )
    
        frame.set_class_costs(np.array([-1, 0, 0, 0, 0]))
        frame.set_class_colors((np.array([
            [0, 0, 0],
            [128, 128, 128],
            [0, 0, 255],
            [255, 255, 255],
            [255, 0, 0]
        ])))
        frame.set_frame_data(img)

        frame.process_safe_distance_zone(min_distance=(2,2), compute_vectorized=False)
        frame.process_distance_to_goal(goal.x, goal.z)

        max_steering = angle.new_deg(40)
        t = math.tan(max_steering.rad())
        L = 4.0
        max_curvature = 2 * t / (L * math.sqrt(4  + t))
        print (f"Max curvature: {max_curvature}")   

        path = HermiteCurve.interpolate(frame.width(), frame.height(), start, goal, max_curvature=max_curvature)
        if path is None:
            print ("Could not interpolate")
            return

        if show_path:
            self.show_curve(frame, path)    


        if not frame.check_feasible_path(min_distance=(2,2), path=path):
            print ("Path not feasible")
            return False

        return True

        
    # def test_connection_1(self):
    #     start = Waypoint(416, 686, angle.new_deg(90 + -0.039754376))
    #     goal = Waypoint(296, 15, angle.new_deg(-8.9513413283239596))
    #     self.intepolate(start, goal)

    # def test_connection_1(self):
    #     start = Waypoint(336, 123, angle.new_deg(-11.9513413283239596))
    #     goal = Waypoint(296, 15, angle.new_deg(-8.9513413283239596))
    #     self.assertTrue(self.intepolate(start, goal))


    # def test_connection_2(self):
    #     start = Waypoint(336, 123, angle.new_deg(90 + -11.9513413283239596))
    #     goal = Waypoint(296, 15, angle.new_deg(-8.9513413283239596))
    #     self.assertFalse(self.intepolate(start, goal))

    # def test_connection_3(self):
    #     start = Waypoint(487, 586, angle.new_deg(-14))
    #     goal = Waypoint(296, 15, angle.new_deg(-20))
    #     self.assertTrue(self.intepolate(start, goal))

    # def test_connection_4(self):
    #     start = Waypoint(486, 679, angle.new_deg(8))
    #     goal = Waypoint(296, 15, angle.new_deg(-8))
    #     self.assertTrue(self.intepolate(start, goal))

    def test_max_curvture(self):
        # start = Waypoint(464, 692, angle.new_deg(90))
        # goal = Waypoint(494, 661, angle.new_deg(25))
        # self.assertTrue(self.intepolate(start, goal))

        start = Waypoint(464, 692, angle.new_deg(90))
        goal = Waypoint(489, 661, angle.new_deg(-35))
        self.assertTrue(self.intepolate(start, goal, show_path=True))


if __name__ == "__main__":
    unittest.main()