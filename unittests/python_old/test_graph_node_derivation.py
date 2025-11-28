import sys
import time
import unittest
import math
from pyfastrrt import CudaGraph
import numpy as np
from pydriveless import SearchFrame, angle, Waypoint, MapPose, WorldPose, CoordinateConverter
import time
from test_utils import TestFrame, TestData, TestUtils, SEGMENTATION_CLASS_COST


def build_test_graph() -> CudaGraph:
    costs = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    return CudaGraph(
        width=256, height=256,
        perception_height_m=256, perception_width_m=256,
        max_steering_angle_deg=40,
        vehicle_length_m=5.412658773,
        min_dist_x=0, min_dist_z=0,
        lower_bound_x=-1, lower_bound_z=-1,
        upper_bound_x=-1, upper_bound_z=-1,
        path_costs=costs
    )


def create_empty_search_frame(w: int, h: int) -> SearchFrame:
    frame = SearchFrame(
        width=w, height=h,
        lower_bound=(-1, -1),
        upper_bound=(-1, -1)
    )
    costs = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    frame.set_class_costs(costs)

    img = np.zeros((h, w, 3), dtype=np.float32)
    frame.set_frame_data(img)
    return frame


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

        graph = build_test_graph()
        graph.add(128, 128, angle.new_deg(0), -1, -1, 0)
        frame = create_empty_search_frame(256, 256)

        graph.derivate_node(frame, angle.new_deg(
            0), PATH_SIZE, VELOCITY, 128, 128)
        graph.accept_derived_nodes()

        if not graph.check_in_graph(128, 78):
            self.fail("position 128, 78 should be in the graph")

        self.assertEqual(graph.get_heading(128, 78), angle.new_deg(0))


    def test_curvy_node_derivation(self):
        location = MapPose(0, 0, 0, angle.new_deg(0))

        graph = build_test_graph()

        path_size = 15
        velocity = 1.0

        ptr = create_empty_search_frame(256, 256)

        for a in range(-180, 180, 18):
            for i in range (-40, 41):
                graph.add(100, 200, angle.new_deg(a), -1, -1, 0)
                graph.derivate_node(ptr,  angle.new_deg(i), path_size, velocity, 100, 200)
                graph.accept_derived_nodes()

                start = Waypoint(100, 200, angle.new_deg(a))
                end = derive(256, 256, 256, 256, location, start, path_size, angle.new_deg(i).rad(), velocity, 0.5 * 5.412658773)

                if not graph.check_in_graph(end[0], end[1]):
                    self.fail(f"pos {end[0]}, {end[1]} should be in graph")

                if graph.get_heading(end[0], end[1]) != angle.new_rad(end[2]):
                    self.fail(f"final heading in {end[0]}, {end[1]} should be {end[2]}")

                graph.clear()

    def test_random_derive(self):
        location = MapPose(0, 0, 0, angle.new_deg(0))

        graph = build_test_graph()

        path_size = 15
        velocity = 1.0

        search_frame = create_empty_search_frame(256, 256)
        pass

if __name__ == "__main__":
    unittest.main()
