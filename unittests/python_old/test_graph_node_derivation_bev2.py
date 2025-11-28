import sys
import time
import unittest
import math
from pyfastrrt import CudaGraph
import numpy as np
from pydriveless import SearchFrame, angle, Waypoint, MapPose, WorldPose, CoordinateConverter
import time, cv2
from test_utils import TestFrame, TestData, TestUtils, SEGMENTATION_CLASS_COST

COSTS = np.array([-1, 0, 0, 0, 0], dtype=np.float32)
COLORS = np.array([
            [0, 0, 0],
            [128, 128, 128],
            [0, 0, 255],
            [255, 255, 255],
            [255, 0, 0]
        ])

def build_test_graph(frame: np.ndarray) -> CudaGraph:
    W = frame.shape[1]
    H = frame.shape[0]
    return CudaGraph(
        width=W, height=H,
        perception_height_m=H, perception_width_m=W,
        max_steering_angle_deg=40,
        vehicle_length_m=5.412658773,
        min_dist_x=0, min_dist_z=0,
        lower_bound_x=-1, lower_bound_z=-1,
        upper_bound_x=-1, upper_bound_z=-1,
        path_costs=COSTS
    )

def read_search_data(file: str) -> tuple[SearchFrame, CudaGraph]:
    f = np.array(cv2.imread(file), dtype=np.float32)
    frame = SearchFrame(
        width=f.shape[1], height=f.shape[0],
        lower_bound=(-1, -1),
        upper_bound=(-1, -1)
    )
    frame.set_class_costs(COSTS)
    frame.set_class_colors(COLORS)
    frame.set_frame_data(f)

    graph = build_test_graph(f)

    return frame, graph

def derive(width: int, height: int, width_m: float, height_m: float,
    location: MapPose, start: Waypoint, pathSize: float,steeringAngle: float, velocity_m_s: float, lr: float):

    wp = WorldPose(angle.new_rad(0), angle.new_rad(0), 0.0, angle.new_rad(0))
    conv = CoordinateConverter(wp, width, height, width_m, height_m)

    conv.set_yaypoint_coordinate_origin(128, 128)
    mapStart = conv.convert(location, start)

    steer = math.tan(steeringAngle)
    dt = 0.1
    ds = velocity_m_s * dt
    beta = math.atan(steer / 2)
    heading_increment_factor = ds * math.cos(beta) * steer / (2 * lr)

    x = mapStart.x
    y = mapStart.y
    maxSize = int(round(pathSize)) + 1

    size = 0

    last_x = start.x
    last_z = start.z
    heading = mapStart.heading.rad()
    localHeading = start.heading.rad()

    parentCost = 0.0
    nodeCost = parentCost
    lastp = [0, 0]

    while (size < maxSize):
        x += ds * math.cos(heading + beta)
        y += ds * math.sin(heading + beta)
        heading += heading_increment_factor

        wpp = MapPose(x, y, 0, angle.new_rad(heading))
        p = conv.convert(location, wpp)

        lastp[0] = p.x
        lastp[1] = p.z
        localHeading = p.heading.rad()
        if lastp[0] == last_x and lastp[1] == last_z:
            continue

        if lastp[0] < 0 or lastp[0] >= width:
            break

        if lastp[1] < 0 or lastp[1] >= height:
            break

        size += 1
        nodeCost += 1
        last_x = lastp[0]
        last_z = lastp[1]

    return (last_x, last_z, localHeading)



class TestCudaGraphNodeDerivation(unittest.TestCase):

    def test_straight(self):
        PATH_SIZE = 49
        VELOCITY = 1.0
        STEERING = 0.0

        frame, graph = read_search_data("bev_2.png")

        p = (416, 686, angle.new_deg(90))
        g = (416+49+1, 686, angle.new_deg(90))

        graph.add(p[0], p[1], p[2], -1, -1, 0)

        graph.derivate_node(frame, angle.new_deg(STEERING), PATH_SIZE, VELOCITY, p[0], p[1])
        graph.accept_derived_nodes()

        #TestUtils.output_graph_nodes(frame, graph, "output1.png")

        if not graph.check_in_graph(g[0], g[1]):
            self.fail(f"position {g[0]}, {g[1]} should be in the graph")

        self.assertEqual(graph.get_heading(g[0], g[1]), g[2])

    # def test_curve_1(self):
    #     PATH_SIZE = 5
    #     VELOCITY = 1.0
    #     STEERING = -0.239456

    #     frame, graph = read_search_data("bev_2.png")

    #     # p = (416, 686, angle.new_deg(90))
    #     # g = (416+49+1, 686, angle.new_deg(90))
    #     p = (416, 686, angle.new_rad(1.570103))
    #     g = (416+49+1, 686, angle.new_deg(90))

    #     graph.add(p[0], p[1], p[2], -1, -1, 0)

    #     graph.derivate_node(frame, angle.new_rad(STEERING), PATH_SIZE, VELOCITY, p[0], p[1])
    #     graph.accept_derived_nodes()

    #     TestUtils.output_graph_nodes(frame, graph, "output1.png")

    #     if not graph.check_in_graph(g[0], g[1]):
    #         self.fail(f"position {g[0]}, {g[1]} should be in the graph")

    #     self.assertEqual(graph.get_heading(g[0], g[1]), g[2])

    
if __name__ == "__main__":
    unittest.main()
