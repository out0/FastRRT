from pydriveless import Waypoint, Interpolator, angle
import numpy as np
import cv2
from hermite import HermiteCurve

def show_curve(frame: np.ndarray, path: np.ndarray, pos: int):
    if frame is None:
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    if pos > len(path) - 2:
        return

    pos1 = pos
    pos2 = pos + 1

    p1 = Waypoint(
        x=int(path[pos1, 0]),
        z=int(path[pos1, 1]),
        heading=angle.new_rad(path[pos1, 2])
    )
    p2 = Waypoint(
        x=int(path[pos2, 0]),
        z=int(path[pos2, 1]),
        heading=angle.new_rad(path[pos2, 2])
    )

    # points = Interpolator.hermite(1000, 1000, p1, p2)
    # for p in points:
    #     frame[p.z, p.x, :] = [255, 255, 255]


    points = HermiteCurve.interpolate(1000, 1000, (p1.x, p1.z, p1.heading.rad()), (p2.x, p2.z, p2.heading.rad()), max_curvature=0.63)
    if points is None:
        print ("Could not interpolate")
        return
    
    for p in points:
        frame[p[1], p[0], :] = [255, 255, 255]

    cv2.imwrite("output1.png", frame)

def main():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    path = np.load("optimized_path.npy")
    for i in  range(0, len(path) - 1):
        show_curve(frame, path, i)
        #input("press enter")

    

   

    





if __name__ == "__main__":
    main()