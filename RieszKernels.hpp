#pragma once


double compute_coulomb_energy(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
);


double compute_coulomb_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
);


double constrain_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
);


double move_points(
    double *__restrict__ points_x,
    double *__restrict__ points_y,
    double *__restrict__ points_z,
    const double *__restrict__ step_x,
    const double *__restrict__ step_y,
    const double *__restrict__ step_z,
    double step_size,
    int num_points
);


double move_points(
    double *__restrict__ new_points_x,
    double *__restrict__ new_points_y,
    double *__restrict__ new_points_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    const double *__restrict__ step_x,
    const double *__restrict__ step_y,
    const double *__restrict__ step_z,
    double step_size,
    int num_points
);


double dot_product(
    const double *__restrict__ vx,
    const double *__restrict__ vy,
    const double *__restrict__ vz,
    const double *__restrict__ wx,
    const double *__restrict__ wy,
    const double *__restrict__ wz,
    int num_points
);


void xpay(
    double *__restrict__ vx,
    double *__restrict__ vy,
    double *__restrict__ vz,
    double a,
    const double *__restrict__ wx,
    const double *__restrict__ wy,
    const double *__restrict__ wz,
    int num_points
);
