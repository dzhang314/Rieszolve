#include <cmath>


static inline double compute_coulomb_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    using std::sqrt;
    double energy = 0.0;
#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (int i = 0; i < num_points; i++) {
        const double xi = points_x[i];
        const double yi = points_y[i];
        const double zi = points_z[i];
        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
#pragma omp simd reduction(+ : energy, fx, fy, fz) simdlen(8)                  \
    aligned(forces_x, forces_y, forces_z, points_x, points_y, points_z : 64)
        for (int j = 0; j < num_points; j++) {
            if (i != j) {
                const double delta_x = xi - points_x[j];
                const double delta_y = yi - points_y[j];
                const double delta_z = zi - points_z[j];
                const double norm_squared =
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                const double inv_norm = 1.0 / sqrt(norm_squared);
                energy += inv_norm;
                const double inv_norm_cubed = inv_norm / norm_squared;
                fx += delta_x * inv_norm_cubed;
                fy += delta_y * inv_norm_cubed;
                fz += delta_z * inv_norm_cubed;
            }
        }
        forces_x[i] = fx;
        forces_y[i] = fy;
        forces_z[i] = fz;
    }
    return energy;
}


static inline double constrain_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    using std::sqrt;
    double force_norm_squared = 0.0;
#pragma omp simd reduction(+ : force_norm_squared) simdlen(8)                  \
    aligned(forces_x, forces_y, forces_z, points_x, points_y, points_z : 64)
    for (int i = 0; i < num_points; i++) {
        const double fx = forces_x[i];
        const double fy = forces_y[i];
        const double fz = forces_z[i];
        const double x = points_x[i];
        const double y = points_y[i];
        const double z = points_z[i];
        const double dot = fx * x + fy * y + fz * z;
        const double proj_x = fx - dot * x;
        const double proj_y = fy - dot * y;
        const double proj_z = fz - dot * z;
        forces_x[i] = proj_x;
        forces_y[i] = proj_y;
        forces_z[i] = proj_z;
        force_norm_squared +=
            proj_x * proj_x + proj_y * proj_y + proj_z * proj_z;
    }
    return sqrt(force_norm_squared);
}


static inline void move_points(
    double *__restrict__ points_x,
    double *__restrict__ points_y,
    double *__restrict__ points_z,
    const double *__restrict__ step_x,
    const double *__restrict__ step_y,
    const double *__restrict__ step_z,
    double step_size,
    int num_points
) {
    using std::sqrt;
#pragma omp simd simdlen(8)                                                    \
    aligned(points_x, points_y, points_z, step_x, step_y, step_z : 64)
    for (int i = 0; i < num_points; i++) {
        const double x = points_x[i] + step_size * step_x[i];
        const double y = points_y[i] + step_size * step_y[i];
        const double z = points_z[i] + step_size * step_z[i];
        const double norm_squared = x * x + y * y + z * z;
        const double inv_norm = 1.0 / sqrt(norm_squared);
        points_x[i] = x * inv_norm;
        points_y[i] = y * inv_norm;
        points_z[i] = z * inv_norm;
    }
}
