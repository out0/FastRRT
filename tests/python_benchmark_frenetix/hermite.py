import numpy as np
import math

def conv_angle(a: float) -> float:
    d = math.degrees(a)
    if d > 180:
        return 360 - d
    if d < -180:
        return 360 + d
    return d

class HermiteCurve:
    
    HALF_PI = 0.5*math.pi

    def distance_between(p1: tuple[int, int, float], p2: tuple[int, int, float]):
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        return math.sqrt(dx ** 2 + dz ** 2)
    


    def interpolate(width: int, height: int, p1: tuple[int, int, float], p2: tuple[int, int, float], max_curvature: float = 0.30) -> list[tuple[int, int, float]] | None:

        dist = HermiteCurve.distance_between(p1, p2)
        num_points = int(round(dist))

        a1 = p1[2] - HermiteCurve.HALF_PI
        a2 = p2[2] - HermiteCurve.HALF_PI

        tan1 = (dist * math.cos(a1), dist * math.sin(a1))
        tan2 = (dist * math.cos(a2), dist * math.sin(a2))
        
        lastp = (-1, -1)

        curve = []

        max_k = -1

        for i in range(0, num_points):
            t = i / (num_points - 1)
            t2 = t ** 2
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            x = h00 * p1[0] + h10 * tan1[0] + h01 * p2[0] + h11 * tan2[0]
            z = h00 * p1[1] + h10 * tan1[1] + h01 * p2[1] + h11 * tan2[1]

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

            ddx = t00 * p1[0] + t10 * tan1[0] + t01 * p2[0] + t11 * tan2[0]
            ddz = t00 * p1[1] + t10 * tan1[1] + t01 * p2[1] + t11 * tan2[1]

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


            dd2x = d00 * p1[0] + d10 * tan1[0] + d01 * p2[0] + d11 * tan2[0];
            dd2z = d00 * p1[1] + d10 * tan1[1] + d01 * p2[1] + d11 * tan2[1];
            k = abs(ddx * dd2z - ddz * dd2x) / math.pow(ddx * ddx + ddz * ddz, 3 / 2)
            max_k = max(max_k, k)

            if (max_curvature > 0 and k > max_curvature):
                print(f"the curvature {k} exceeds the maximum value {max_curvature}")
                #return None


            curve.append((cx, cz, heading))
            lastp = (cx, cz)

        print(f"curve {p1} --> {p2} has max curvature of {max_k}")
        return curve