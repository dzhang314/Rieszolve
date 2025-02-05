#pragma once


void triangulate(
    int *__restrict__ faces,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
);


void convex_hull(
    int *__restrict__ faces,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
);
