import numpy as np
import math
from pydriveless import angle, Waypoint

def conv_angle(a: float) -> float:
    d = math.degrees(a)
    if d > 180:
        return 360 - d
    if d < -180:
        return 360 + d
    return d

class HermiteCurve:
    
    HALF_PI = 0.5*math.pi

    def distance_between(p1: Waypoint, p2: Waypoint):
        dx = p2.x - p1.x
        dz = p2.z - p1.z
        return math.sqrt(dx ** 2 + dz ** 2)
    


    def interpolate(width: int, height: int, p1: Waypoint, p2: Waypoint, max_curvature: float = 0.30) -> list[tuple[int, int, float]] | None:

        dist = HermiteCurve.distance_between(p1, p2)
        num_points = int(round(dist))

        a1 = p1.heading.rad() - HermiteCurve.HALF_PI
        a2 = p2.heading.rad() - HermiteCurve.HALF_PI

        tan1 = (dist * math.cos(a1), dist * math.sin(a1))
        tan2 = (dist * math.cos(a2), dist * math.sin(a2))
        
        lastp = (-1, -1)

        curve = []

        for i in range(0, num_points):
            t = i / (num_points - 1)
            t2 = t ** 2
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            x = h00 * p1.x + h10 * tan1[0] + h01 * p2.x + h11 * tan2[0]
            z = h00 * p1.z + h10 * tan1[1] + h01 * p2.z + h11 * tan2[1]

            if x < 0 or x >= width:
                continue
            if z < 0 or z >= height:
                continue

            cx = int(round(x))
            cz = int(round(z))

            if cx == lastp[0] and cz == lastp[1]:
                continue
            if cx < 0 or cx >= width:
                continue
            if cz < 0 or cz >= height:
                continue

            t00 = 6 * t2 - 6 * t
            t10 = 3 * t2 - 4 * t + 1
            t01 = -6 * t2 + 6 * t
            t11 = 3 * t2 - 2 * t

            ddx = t00 * p1.x + t10 * tan1[0] + t01 * p2.x + t11 * tan2[0]
            ddz = t00 * p1.z + t10 * tan1[1] + t01 * p2.z + t11 * tan2[1]

            heading = math.atan2(ddz, ddx) + HermiteCurve.HALF_PI

            # #print (f"heading: {math.degrees(heading)}")
            # hv = False
            # if 2*(heading - last_heading) > math.radians(40):
            #     print(f"heading violation in {cx}, {cz} interpolating {p1} -> {p2}:  old: {conv_angle(last_heading)} x  new: {conv_angle(heading)}:  {conv_angle(heading) - conv_angle(last_heading)} difference")
            #     hv = True
            
            d00 = 12 * t - 6
            d10 = 6 * t - 4
            d01 = -12 * t + 6
            d11 = 6 * t - 2


            dd2x = d00 * p1.x + d10 * tan1[0] + d01 * p2.x + d11 * tan2[0]
            dd2z = d00 * p1.z + d10 * tan1[1] + d01 * p2.z + d11 * tan2[1]
        
            if (max_curvature > 0):
                k = abs(ddx * dd2z - ddz * dd2x) / math.pow(ddx * ddx + ddz * ddz, 3 / 2)
                print (f"Curvature at {cx}, {cz} : {k}")
                if (k > max_curvature):
                    return None

            curve.append(Waypoint(cx, cz, angle.new_rad(heading)))
            lastp = (cx, cz)

        return curve