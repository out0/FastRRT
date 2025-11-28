#include "../include/fastrrt.h"
#include <driveless/cuda_params.h>
#include <driveless/math_utils.h>
#include <driveless/waypoint.h>
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

int max(int a, int b, int c)
{
    int v = (a > b) ? a : b;
    return (v > c) ? v : c;
}

std::vector<Waypoint> interpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2)
{
    std::vector<Waypoint> curve;

    // int numPoints = 2 * abs(max(int(p2.z() - p1.z()), int(p2.x() - p1.x()), 100));

    int numPoints = TO_INT(Waypoint::distanceBetween(p1, p2));

    // int numPoints = 100;
    curve.reserve(numPoints);

    // Distance between points (used to scale tangents)
    float dx = p2.x() - p1.x();
    float dz = p2.z() - p1.z();
    float d = sqrtf(dx * dx + dz * dz);

    float a1 = p1.heading().rad() - PI / 2;
    float a2 = p2.heading().rad() - PI / 2;

    // Tangent vectors
    float2 tan1 = {d * cosf(a1), d * sinf(a1)};
    float2 tan2 = {d * cosf(a2), d * sinf(a2)};

    int last_x = -1;
    int last_z = -1;

    for (int i = 0; i < numPoints; ++i)
    {
        float t = static_cast<float>(i) / (numPoints - 1);

        float t2 = t * t;
        float t3 = t2 * t;

        // Hermite basis functions
        float h00 = 2 * t3 - 3 * t2 + 1;
        float h10 = t3 - 2 * t2 + t;
        float h01 = -2 * t3 + 3 * t2;
        float h11 = t3 - t2;

        float x = h00 * p1.x() + h10 * tan1.x + h01 * p2.x() + h11 * tan2.x;
        float z = h00 * p1.z() + h10 * tan1.y + h01 * p2.z() + h11 * tan2.y;

        if (x < 0 || x >= width)
            continue;
        if (z < 0 || z >= height)
            continue;

        int cx = static_cast<int>(round(x));
        int cz = static_cast<int>(round(z));

        if (cx == last_x && cz == last_z)
            continue;
        if (cx < 0 || cx >= width)
            continue;
        if (cz < 0 || cz >= height)
            continue;

        float t00 = 6 * t2 - 6 * t;
        float t10 = 3 * t2 - 4 * t + 1;
        float t01 = -6 * t2 + 6 * t;
        float t11 = 3 * t2 - 2 * t;

        float ddx = t00 * p1.x() + t10 * tan1.x + t01 * p2.x() + t11 * tan2.x;
        float ddz = t00 * p1.z() + t10 * tan1.y + t01 * p2.z() + t11 * tan2.y;

        float heading = atan2f(ddz, ddx) + HALF_PI;

        // Interpolated point
        curve.push_back({cx, cz, angle::rad(heading)});
        last_x = cx;
        last_z = cz;
    }

    return curve;
}

std::vector<Waypoint> interpolate(std::vector<Waypoint> &controlPoints, int width, int height)
{
    int resolution = 64;
    std::vector<Waypoint> interpolatedPoints;

    if (controlPoints.size() < 2)
    {
        printf("At least 2 control points are required for hermite interpolation.\n");
        return {};
    }

    Waypoint p1 = controlPoints[0];

    for (size_t i = 1; i < controlPoints.size(); i++)
    {
        Waypoint p2 = controlPoints[i];
        //printf ("interpolating %d, %d, %f --> %d, %d, %f\n", p1.x(), p1.z(), p1.heading().deg(), p2.x(), p2.z(), p2.heading().deg());
        std::vector<Waypoint> partialCurve = interpolateHermiteCurve(width, height, p1, p2);
        interpolatedPoints.insert(interpolatedPoints.end(), partialCurve.begin(), partialCurve.end());
        p1 = p2;
    }

    return interpolatedPoints;
}
