#include <cmath>


static inline void two_sum(double &s, double &e, double x, double y) {
    s = x + y;
    const double x_eff = s - y;
    const double y_eff = s - x_eff;
    const double x_err = x - x_eff;
    const double y_err = y - y_eff;
    e = x_err + y_err;
}


static inline void two_sum(double &s, double &e) {
    const double x = s;
    const double y = e;
    two_sum(s, e, x, y);
}


struct HighPrecisionAccumulator {

    double terms[2];

    constexpr HighPrecisionAccumulator() noexcept
        : terms{0.0, 0.0} {}

    constexpr double to_double() const noexcept { return terms[0] + terms[1]; }

    constexpr void add(double x) noexcept {
        two_sum(terms[0], x);
        two_sum(terms[1], x);
    }

}; // struct HighPrecisionAccumulator


static inline double compute_coulomb_energy(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    using std::sqrt;
    HighPrecisionAccumulator energy;
    for (int i = 0; i < num_points; ++i) {
        const double xi = points_x[i];
        const double yi = points_y[i];
        const double zi = points_z[i];
        for (int j = 0; j < num_points; ++j) {
            if (i != j) {
                const double delta_x = xi - points_x[j];
                const double delta_y = yi - points_y[j];
                const double delta_z = zi - points_z[j];
                const double norm_squared =
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                energy.add(1.0 / sqrt(norm_squared));
            }
        }
    }
    return energy.to_double();
}


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
    HighPrecisionAccumulator energy;
    for (int i = 0; i < num_points; ++i) {
        const double xi = points_x[i];
        const double yi = points_y[i];
        const double zi = points_z[i];
        HighPrecisionAccumulator fx;
        HighPrecisionAccumulator fy;
        HighPrecisionAccumulator fz;
        for (int j = 0; j < num_points; ++j) {
            if (i != j) {
                const double delta_x = xi - points_x[j];
                const double delta_y = yi - points_y[j];
                const double delta_z = zi - points_z[j];
                const double norm_squared =
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                const double inv_norm = 1.0 / sqrt(norm_squared);
                energy.add(inv_norm);
                const double inv_norm_cubed = inv_norm / norm_squared;
                fx.add(delta_x * inv_norm_cubed);
                fy.add(delta_y * inv_norm_cubed);
                fz.add(delta_z * inv_norm_cubed);
            }
        }
        forces_x[i] = fx.to_double();
        forces_y[i] = fy.to_double();
        forces_z[i] = fz.to_double();
    }
    return energy.to_double();
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
    HighPrecisionAccumulator force_norm_squared;
    for (int i = 0; i < num_points; ++i) {
        const double fx = forces_x[i];
        const double fy = forces_y[i];
        const double fz = forces_z[i];
        const double x = points_x[i];
        const double y = points_y[i];
        const double z = points_z[i];
        const double dot = std::fma(fx, x, std::fma(fy, y, fz * z));
        const double proj_x = -std::fma(dot, x, -fx);
        const double proj_y = -std::fma(dot, y, -fy);
        const double proj_z = -std::fma(dot, z, -fz);
        forces_x[i] = proj_x;
        forces_y[i] = proj_y;
        forces_z[i] = proj_z;
        force_norm_squared.add(
            proj_x * proj_x + proj_y * proj_y + proj_z * proj_z
        );
    }
    return sqrt(force_norm_squared.to_double());
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
    for (int i = 0; i < num_points; ++i) {
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


static inline void move_points(
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
) {
    using std::sqrt;
#pragma omp simd simdlen(8)                                                    \
    aligned(new_points_x, new_points_y, new_points_z : 64)                     \
    aligned(points_x, points_y, points_z, step_x, step_y, step_z : 64)
    for (int i = 0; i < num_points; ++i) {
        const double x = points_x[i] + step_size * step_x[i];
        const double y = points_y[i] + step_size * step_y[i];
        const double z = points_z[i] + step_size * step_z[i];
        const double norm_squared = x * x + y * y + z * z;
        const double inv_norm = 1.0 / sqrt(norm_squared);
        new_points_x[i] = x * inv_norm;
        new_points_y[i] = y * inv_norm;
        new_points_z[i] = z * inv_norm;
    }
}
