#pragma once


double compute_coulomb_energy(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
);


void compute_coulomb_forces(
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


double compute_step_direction(
    double *__restrict__ step_x,
    double *__restrict__ step_y,
    double *__restrict__ step_z,
    const double *__restrict__ forces_x,
    const double *__restrict__ forces_y,
    const double *__restrict__ forces_z,
    const double *__restrict__ prev_forces_x,
    const double *__restrict__ prev_forces_y,
    const double *__restrict__ prev_forces_z,
    int num_points,
    bool conjugate_gradient
);


double quadratic_line_search(
    double *__restrict__ points_x,
    double *__restrict__ points_y,
    double *__restrict__ points_z,
    double *__restrict__ temp_points_x,
    double *__restrict__ temp_points_y,
    double *__restrict__ temp_points_z,
    double &step_size,
    const double *__restrict__ step_x,
    const double *__restrict__ step_y,
    const double *__restrict__ step_z,
    double initial_energy,
    int num_points
);
