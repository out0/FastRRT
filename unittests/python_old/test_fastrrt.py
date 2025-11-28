import unittest, math, numpy as np
from pydriveless import angle
from pydriveless import MapPose
from pydriveless import SearchFrame
from pyfastrrt import FastRRT
from test_utils import *
import cv2
import_cv2()

class TestFastRRT(unittest.TestCase):

    def test_fast_rrt(self):
        img = np.array(cv2.imread("../bev_1.png"), dtype=np.float32)


        cframe = SearchFrame(
            width=img.shape[1],
            height=img.shape[0],
            lower_bound=EGO_LOWER_BOUND,
            upper_bound=EGO_UPPER_BOUND
        )

        cframe.set_frame_data(img)
        cframe.set_class_costs(SEGMENTATION_CLASS_COST)
        cframe.set_class_colors(SEGMENTED_COLORS)

        i = 1
        planner = FastRRT(
            search_frame=cframe,
            perception_width_m=OG_REAL_WIDTH,
            perception_height_m=OG_REAL_HEIGHT,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=-1,
            min_dist_x=MIN_DISTANCE_WIDTH_PX,
            min_dist_z=MIN_DISTANCE_HEIGHT_PX,
            path_costs=SEGMENTATION_CLASS_COST
        )
        
        planner.set_plan_data(
            cuda_ptr=cframe,
            start=(128, 128, 0.0),
            goal=(128, 0, 0.0),
            velocity_m_s=1
        )

        cframe.process_safe_distance_zone((MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX), False)
        cframe.process_distance_to_goal(128, 0)
        planner.search_init()

        while not planner.goal_reached():
            planner.loop(False)

        planner.path_optimize()

        path = planner.get_planned_path(True)

        f = cframe.get_color_frame()
        for i in range(path.shape[0]):
            f[int(path[i, 1]), int(path[i, 0]), :] = [255, 255, 255]

        cv2.imwrite("output1.png", f)

if __name__ == "__main__":
    unittest.main()
