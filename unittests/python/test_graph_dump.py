import unittest, math, numpy as np
from pydriveless import angle
from pydriveless import MapPose
from pydriveless import SearchFrame
from pyfastrrt import CudaGraph
from hermite import HermiteCurve
from test_utils import *
import cv2
import_cv2()

class TestGraphDump(unittest.TestCase):

    def test_graph_dump(self):
        g = CudaGraph(100, 100, 10, 10, 40, 4, 0, 0, -1, -1, -1, -1, np.zeros((5)), -1)
        g2 = CudaGraph(100, 100, 10, 10, 40, 4, 0, 0, -1, -1, -1, -1, np.zeros((5)), -1)
        
        g.add(50, 50, angle.new_deg(0), -1, -1, 0)
        g.add(40, 40, angle.new_deg(-11), 50, 50, 10)
        g.add_temporary(30, 30, angle.new_deg(4), 40, 40, -20)
        
        g.dump_graph_to_file("test.dat")
        g.clear()
        self.assertEqual(0, g.count_all())

        g2.read_from_dump_file("test.dat")
        self.assertEqual(3, g2.count_all())

        nodes = g2.list_all()

        found = [0, 0, 0]

        for n in nodes:
            nx, ny, _, nz = n
            if nx == 50:
                found[0] += 1
                self.assertEqual(nz, GRAPH_TYPE_NODE)
                px, py = g2.get_parent(int(nx), int(ny))
                self.assertEqual(-1, px)
                self.assertEqual(-1, py)
                self.assertEqual(0, g2.get_cost(int(nx), int(ny)))
            elif nx == 40:
                found[1] += 1
                self.assertEqual(nz, GRAPH_TYPE_NODE)
                px, py = g2.get_parent(int(nx), int(ny))
                self.assertEqual(50, px)
                self.assertEqual(50, py)
                self.assertEqual(10, g2.get_cost(int(nx), int(ny)))
            elif nx == 30:
                found[2] += 1
                self.assertEqual(nz, GRAPH_TYPE_TEMP)
                px, py = g2.get_parent(int(nx), int(ny))
                self.assertEqual(40, px)
                self.assertEqual(40, py)
                self.assertEqual(-20, g2.get_cost(int(nx), int(ny)))
    
            self.assertEqual(nx, ny)

        self.assertEqual(1, found[0])
        self.assertEqual(1, found[1])
        self.assertEqual(1, found[2])
        
        os.remove("test.dat")

if __name__ == "__main__":
    unittest.main()