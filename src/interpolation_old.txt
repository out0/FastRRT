#include "../include/fastrrt.h"
#include "../include/cuda_params.h"
#include "../include/math_utils.h"
#include "../include/waypoint.h"
#include <bits/algorithmfwd.h>

// extern std::vector<Waypoint> drawKinematicPath(double *physicalParams, int width,  int height, int2 center, int2 start, float start_heading, Waypoint end, float velocity_m_s);


// std::vector<Waypoint> interpolateKinematics(std::vector<Waypoint>& path, double *physicalParams, int width, int height, int2 center, float velocity_m_s) {
//     int2 start = {path[0].x(), path[0].z()};
//     float curr_heading = path[0].heading().rad();

//     std::vector<Waypoint> res;

//     for (int i = 1; i < path.size(); i++) {
//         printf ("start: %d, %d, %f\n", start.x, start.y, curr_heading);
//         std::vector<Waypoint> subpath = drawKinematicPath(physicalParams, width, height, center, start, curr_heading, path[i], velocity_m_s);
//         printf("subpath size: %ld\n", subpath.size());
//         res.insert(res.end(), subpath.begin(), subpath.end());
        
//         Waypoint last = subpath.at(subpath.size() - 1);
//         start.x = last.x();
//         start.y = last.z();
//         curr_heading = last.heading().rad();

//     }
//     return res;
// }


Waypoint evaluateBezier(Waypoint& P0, Waypoint& P1, Waypoint& P2, Waypoint& P3, double t) {
    double x = std::pow(1 - t, 3) * P0.x() + 3 * std::pow(1 - t, 2) * t * P1.x() + 
               3 * (1 - t) * std::pow(t, 2) * P2.x() + std::pow(t, 3) * P3.x();

    double z = std::pow(1 - t, 3) * P0.z() + 3 * std::pow(1 - t, 2) * t * P1.z() + 
               3 * (1 - t) * std::pow(t, 2) * P2.z() + std::pow(t, 3) * P3.z();

    return Waypoint(x, z, angle::rad(0));
}

std::vector<Waypoint> interpolate(std::vector<Waypoint>& controlPoints, int width, int height) {
    int resolution = 64;
    std::vector<Waypoint> interpolatedPoints;
    
    if (controlPoints.size() < 4) {
        printf("At least 4 control points are required for cubic Bezier interpolation.\n");
        return {};
    }

    for (size_t i = 0; i + 3 < controlPoints.size(); i += 3) {
        for (int j = 0; j <= resolution; ++j) {
            double t = static_cast<double>(j) / resolution;
            Waypoint p = evaluateBezier(controlPoints[i], controlPoints[i + 1], 
                controlPoints[i + 2], controlPoints[i + 3], t);

            if (p.x() < 0 || p.x() > width) continue;
            if (p.z() < 0 || p.z() > height) continue;
            interpolatedPoints.push_back(p);
        }
    }

    return interpolatedPoints;
}
std::vector<Waypoint> interpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2) {
    std::vector<Waypoint> curve;
    
    int numPoints = 2*abs(int(p2.z() - p1.z()));
    //int numPoints = 100;
    curve.reserve(numPoints);

    // Distance between points (used to scale tangents)
    float dx = p2.x() - p1.x();
    float dz = p2.z() - p1.z();
    float d = sqrtf(dx * dx + dz * dz);

    float a1 = p1.heading().rad() - PI/2;
    float a2 = p2.heading().rad() - PI/2;

    // Tangent vectors
    float2 tan1 = {  d * cosf(a1), d * sinf(a1) };
    float2 tan2 = {  d * cosf(a2), d * sinf(a2) };

    int last_x = -1;
    int last_z = -1;

    for (int i = 0; i < numPoints; ++i) {
        float t = static_cast<float>(i) / (numPoints - 1);

        float t2 = t*t;
        float t3 = t2*t;

        // Hermite basis functions
        float h00 =  2 * t3 - 3 * t2 + 1;
        float h10 =      t3 - 2 * t2 + t;
        float h01 = -2 * t3 + 3 * t2;
        float h11 =      t3     - t2;

        float x = h00 * p1.x() + h10 * tan1.x + h01 * p2.x() + h11 * tan2.x;
        float z = h00 * p1.z() + h10 * tan1.y + h01 * p2.z() + h11 * tan2.y;

        if (x < 0 || x >= width) continue;
        if (z < 0 || z >= height) continue;

        int cx = static_cast<int>(round(x));
        int cz = static_cast<int>(round(z));

        if (cx == last_x && cz == last_z) continue;

        // Interpolated point
        curve.push_back({cx, cz, angle::rad(0)});
        last_x = cx;
        last_z = cz;
    }

    return curve;
}