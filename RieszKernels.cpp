#include "RieszKernels.hpp"

#include <cmath>
using std::fma;
using std::sqrt;


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

    HighPrecisionAccumulator() noexcept
        : terms{0.0, 0.0} {}

    double to_double() const noexcept { return terms[0] + terms[1]; }

    void add(double x) noexcept {
        two_sum(terms[0], x);
        two_sum(terms[1], x);
    }

    HighPrecisionAccumulator &operator+=(const HighPrecisionAccumulator &other
    ) noexcept {
        add(other.terms[0]);
        add(other.terms[1]);
        return *this;
    }

}; // struct HighPrecisionAccumulator

#ifdef _OPENMP
#pragma omp declare reduction(                                                 \
        + : HighPrecisionAccumulator : omp_out += omp_in                       \
) initializer(omp_priv = HighPrecisionAccumulator{})
#endif // _OPENMP


double compute_coulomb_energy(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    HighPrecisionAccumulator energy;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : energy)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) {
        const double xi = points_x[i];
        const double yi = points_y[i];
        const double zi = points_z[i];
        for (int j = 0; j < num_points; ++j) {
            if (i != j) {
                const double delta_x = xi - points_x[j];
                const double delta_y = yi - points_y[j];
                const double delta_z = zi - points_z[j];
                const double dist_squared =
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                const double inv_dist = 1.0 / sqrt(dist_squared);
                energy.add(inv_dist);
            }
        }
    }
    return energy.to_double();
}


double compute_coulomb_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    HighPrecisionAccumulator energy;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : energy)
#endif // _OPENMP
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
                const double dist_squared =
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                const double inv_dist = 1.0 / sqrt(dist_squared);
                energy.add(inv_dist);
                const double inv_dist_cubed = inv_dist / dist_squared;
                fx.add(delta_x * inv_dist_cubed);
                fy.add(delta_y * inv_dist_cubed);
                fz.add(delta_z * inv_dist_cubed);
            }
        }
        forces_x[i] = fx.to_double();
        forces_y[i] = fy.to_double();
        forces_z[i] = fz.to_double();
    }
    return energy.to_double();
}


double constrain_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    HighPrecisionAccumulator force_norm_squared;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : force_norm_squared)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) {
        const double fx = forces_x[i];
        const double fy = forces_y[i];
        const double fz = forces_z[i];
        const double x = points_x[i];
        const double y = points_y[i];
        const double z = points_z[i];
        const double dot = fx * x + fy * y + fz * z;
        const double proj_x = -fma(dot, x, -fx);
        const double proj_y = -fma(dot, y, -fy);
        const double proj_z = -fma(dot, z, -fz);
        forces_x[i] = proj_x;
        forces_y[i] = proj_y;
        forces_z[i] = proj_z;
        force_norm_squared.add(
            proj_x * proj_x + proj_y * proj_y + proj_z * proj_z
        );
    }
    return force_norm_squared.to_double();
}


double move_points(
    double *__restrict__ points_x,
    double *__restrict__ points_y,
    double *__restrict__ points_z,
    const double *__restrict__ step_x,
    const double *__restrict__ step_y,
    const double *__restrict__ step_z,
    double step_size,
    int num_points
) {
    HighPrecisionAccumulator distance_squared;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : distance_squared)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) {
        const double x_old = points_x[i];
        const double y_old = points_y[i];
        const double z_old = points_z[i];
        const double x_moved = fma(step_size, step_x[i], x_old);
        const double y_moved = fma(step_size, step_y[i], y_old);
        const double z_moved = fma(step_size, step_z[i], z_old);
        const double norm_squared =
            x_moved * x_moved + y_moved * y_moved + z_moved * z_moved;
        const double inv_norm = 1.0 / sqrt(norm_squared);
        const double x_new = inv_norm * x_moved;
        const double y_new = inv_norm * y_moved;
        const double z_new = inv_norm * z_moved;
        points_x[i] = x_new;
        points_y[i] = y_new;
        points_z[i] = z_new;
        const double delta_x = x_new - x_old;
        const double delta_y = y_new - y_old;
        const double delta_z = z_new - z_old;
        distance_squared.add(
            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        );
    }
    return distance_squared.to_double();
}


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
) {
    HighPrecisionAccumulator distance_squared;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : distance_squared)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) {
        const double x_old = points_x[i];
        const double y_old = points_y[i];
        const double z_old = points_z[i];
        const double x_moved = fma(step_size, step_x[i], x_old);
        const double y_moved = fma(step_size, step_y[i], y_old);
        const double z_moved = fma(step_size, step_z[i], z_old);
        const double norm_squared =
            x_moved * x_moved + y_moved * y_moved + z_moved * z_moved;
        const double inv_norm = 1.0 / sqrt(norm_squared);
        const double x_new = inv_norm * x_moved;
        const double y_new = inv_norm * y_moved;
        const double z_new = inv_norm * z_moved;
        new_points_x[i] = x_new;
        new_points_y[i] = y_new;
        new_points_z[i] = z_new;
        const double delta_x = x_new - x_old;
        const double delta_y = y_new - y_old;
        const double delta_z = z_new - z_old;
        distance_squared.add(
            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        );
    }
    return distance_squared.to_double();
}


double dot_product(
    const double *__restrict__ vx,
    const double *__restrict__ vy,
    const double *__restrict__ vz,
    const double *__restrict__ wx,
    const double *__restrict__ wy,
    const double *__restrict__ wz,
    int num_points
) {
    HighPrecisionAccumulator result;
    for (int i = 0; i < num_points; ++i) { result.add(vx[i] * wx[i]); }
    for (int i = 0; i < num_points; ++i) { result.add(vy[i] * wy[i]); }
    for (int i = 0; i < num_points; ++i) { result.add(vz[i] * wz[i]); }
    return result.to_double();
}


void xpay(
    double *__restrict__ vx,
    double *__restrict__ vy,
    double *__restrict__ vz,
    double a,
    const double *__restrict__ wx,
    const double *__restrict__ wy,
    const double *__restrict__ wz,
    int num_points
) {
#ifdef _OPENMP
#pragma omp simd simdlen(8) aligned(vx, wx : 64)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) { vx[i] = fma(a, vx[i], wx[i]); }
#ifdef _OPENMP
#pragma omp simd simdlen(8) aligned(vy, wy : 64)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) { vy[i] = fma(a, vy[i], wy[i]); }
#ifdef _OPENMP
#pragma omp simd simdlen(8) aligned(vz, wz : 64)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) { vz[i] = fma(a, vz[i], wz[i]); }
}
