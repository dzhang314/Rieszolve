#include "RieszolveKernels.hpp"

#include <cmath>
using std::fma;
using std::sqrt;

#ifndef RIESZOLVE_DISABLE_AVX512
#ifdef __AVX512F__
#define RIESZOLVE_USE_AVX512
#endif // __AVX512F__
#endif // RIESZOLVE_DISABLE_AVX512

#ifdef RIESZOLVE_USE_AVX512
#include <immintrin.h>
#endif // RIESZOLVE_USE_AVX512


static inline void two_sum(double &s, double &e, double x, double y) noexcept {
    s = x + y;
    const double x_eff = s - y;
    const double y_eff = s - x_eff;
    const double x_err = x - x_eff;
    const double y_err = y - y_eff;
    e = x_err + y_err;
}


#ifdef RIESZOLVE_USE_AVX512
static inline void two_sum(__m512d &s, __m512d &e, __m512d x, __m512d y) {
    s = _mm512_add_pd(x, y);
    const __m512d x_eff = _mm512_sub_pd(s, y);
    const __m512d y_eff = _mm512_sub_pd(s, x_eff);
    const __m512d x_err = _mm512_sub_pd(x, x_eff);
    const __m512d y_err = _mm512_sub_pd(y, y_eff);
    e = _mm512_add_pd(x_err, y_err);
}
#endif // RIESZOLVE_USE_AVX512


static inline void two_sum(double &s, double &e) noexcept {
    const double x = s;
    const double y = e;
    two_sum(s, e, x, y);
}


#ifdef RIESZOLVE_USE_AVX512
static inline void two_sum(__m512d &s, __m512d &e) {
    const __m512d x = s;
    const __m512d y = e;
    two_sum(s, e, x, y);
}
#endif // RIESZOLVE_USE_AVX512


struct HighPrecisionAccumulator {

    double terms[2];

    HighPrecisionAccumulator() noexcept
        : terms{0.0, 0.0} {}

    double to_double() const noexcept { return terms[0] + terms[1]; }

    void add(double x) noexcept {
        two_sum(terms[0], x);
        two_sum(terms[1], x);
    }

    HighPrecisionAccumulator &
    operator+=(const HighPrecisionAccumulator &other) noexcept {
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


#ifdef RIESZOLVE_USE_AVX512

struct HighPrecisionVectorAccumulator {

    __m512d terms[2];

    HighPrecisionVectorAccumulator() noexcept
        : terms{_mm512_setzero_pd(), _mm512_setzero_pd()} {}

    double to_double() const noexcept {
        return _mm512_reduce_add_pd(_mm512_add_pd(terms[0], terms[1]));
    }

    void add(__m512d x) noexcept {
        two_sum(terms[0], x);
        two_sum(terms[1], x);
    }

    HighPrecisionVectorAccumulator &
    operator+=(const HighPrecisionVectorAccumulator &other) noexcept {
        add(other.terms[0]);
        add(other.terms[1]);
        return *this;
    }

}; // struct HighPrecisionVectorAccumulator

#ifdef _OPENMP
#pragma omp declare reduction(                                                 \
        + : HighPrecisionVectorAccumulator : omp_out += omp_in                 \
) initializer(omp_priv = HighPrecisionVectorAccumulator{})
#endif // _OPENMP

#endif // RIESZOLVE_USE_AVX512


double compute_coulomb_energy(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) noexcept {
#ifdef RIESZOLVE_USE_AVX512
    const __m512d ONE_VECTOR = _mm512_set1_pd(1.0);
    HighPrecisionVectorAccumulator energy_vector;
    HighPrecisionAccumulator energy_scalar;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : energy_vector, energy_scalar)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) {
        const double xi = points_x[i];
        const double yi = points_y[i];
        const double zi = points_z[i];
        const __m512d xi_vector = _mm512_set1_pd(xi);
        const __m512d yi_vector = _mm512_set1_pd(yi);
        const __m512d zi_vector = _mm512_set1_pd(zi);
        for (int j = 0; j < num_points;) {
            if (j + 8 <= num_points) {
                if ((j <= i) & (i < j + 8)) {
                    for (int k = 0; k < 8; ++j, ++k) {
                        if (i != j) {
                            const double delta_x = xi - points_x[j];
                            const double delta_y = yi - points_y[j];
                            const double delta_z = zi - points_z[j];
                            const double dist_squared = delta_x * delta_x +
                                                        delta_y * delta_y +
                                                        delta_z * delta_z;
                            energy_scalar.add(1.0 / sqrt(dist_squared));
                        }
                    }
                } else {
                    const __m512d delta_x =
                        _mm512_sub_pd(xi_vector, _mm512_load_pd(points_x + j));
                    const __m512d delta_y =
                        _mm512_sub_pd(yi_vector, _mm512_load_pd(points_y + j));
                    const __m512d delta_z =
                        _mm512_sub_pd(zi_vector, _mm512_load_pd(points_z + j));
                    const __m512d dist_squared = _mm512_add_pd(
                        _mm512_add_pd(
                            _mm512_mul_pd(delta_x, delta_x),
                            _mm512_mul_pd(delta_y, delta_y)
                        ),
                        _mm512_mul_pd(delta_z, delta_z)
                    );
                    energy_vector.add(
                        _mm512_div_pd(ONE_VECTOR, _mm512_sqrt_pd(dist_squared))
                    );
                    j += 8;
                }
            } else {
                if (i != j) {
                    const double delta_x = xi - points_x[j];
                    const double delta_y = yi - points_y[j];
                    const double delta_z = zi - points_z[j];
                    const double dist_squared = delta_x * delta_x +
                                                delta_y * delta_y +
                                                delta_z * delta_z;
                    energy_scalar.add(1.0 / sqrt(dist_squared));
                }
                ++j;
            }
        }
    }
    return energy_vector.to_double() + energy_scalar.to_double();
#else
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
#endif // RIESZOLVE_USE_AVX512
}


double compute_coulomb_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) noexcept {
#ifdef RIESZOLVE_USE_AVX512
    const __m512d ONE_VECTOR = _mm512_set1_pd(1.0);
    HighPrecisionVectorAccumulator energy_vector;
    HighPrecisionAccumulator energy_scalar;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : energy_vector, energy_scalar)
#endif // _OPENMP
    for (int i = 0; i < num_points; ++i) {
        const double xi = points_x[i];
        const double yi = points_y[i];
        const double zi = points_z[i];
        const __m512d xi_vector = _mm512_set1_pd(xi);
        const __m512d yi_vector = _mm512_set1_pd(yi);
        const __m512d zi_vector = _mm512_set1_pd(zi);
        HighPrecisionVectorAccumulator fx_vector;
        HighPrecisionVectorAccumulator fy_vector;
        HighPrecisionVectorAccumulator fz_vector;
        HighPrecisionAccumulator fx_scalar;
        HighPrecisionAccumulator fy_scalar;
        HighPrecisionAccumulator fz_scalar;
        for (int j = 0; j < num_points;) {
            if (j + 8 <= num_points) {
                if ((j <= i) & (i < j + 8)) {
                    for (int k = 0; k < 8; ++j, ++k) {
                        if (i != j) {
                            const double delta_x = xi - points_x[j];
                            const double delta_y = yi - points_y[j];
                            const double delta_z = zi - points_z[j];
                            const double dist_squared = delta_x * delta_x +
                                                        delta_y * delta_y +
                                                        delta_z * delta_z;
                            const double inv_dist = 1.0 / sqrt(dist_squared);
                            energy_scalar.add(inv_dist);
                            const double inv_dist_cubed =
                                inv_dist / dist_squared;
                            fx_scalar.add(delta_x * inv_dist_cubed);
                            fy_scalar.add(delta_y * inv_dist_cubed);
                            fz_scalar.add(delta_z * inv_dist_cubed);
                        }
                    }
                } else {
                    const __m512d delta_x =
                        _mm512_sub_pd(xi_vector, _mm512_load_pd(points_x + j));
                    const __m512d delta_y =
                        _mm512_sub_pd(yi_vector, _mm512_load_pd(points_y + j));
                    const __m512d delta_z =
                        _mm512_sub_pd(zi_vector, _mm512_load_pd(points_z + j));
                    const __m512d dist_squared = _mm512_add_pd(
                        _mm512_add_pd(
                            _mm512_mul_pd(delta_x, delta_x),
                            _mm512_mul_pd(delta_y, delta_y)
                        ),
                        _mm512_mul_pd(delta_z, delta_z)
                    );
                    const __m512d inv_dist =
                        _mm512_div_pd(ONE_VECTOR, _mm512_sqrt_pd(dist_squared));
                    energy_vector.add(inv_dist);
                    const __m512d inv_dist_cubed =
                        _mm512_div_pd(inv_dist, dist_squared);
                    fx_vector.add(_mm512_mul_pd(delta_x, inv_dist_cubed));
                    fy_vector.add(_mm512_mul_pd(delta_y, inv_dist_cubed));
                    fz_vector.add(_mm512_mul_pd(delta_z, inv_dist_cubed));
                    j += 8;
                }
            } else {
                if (i != j) {
                    const double delta_x = xi - points_x[j];
                    const double delta_y = yi - points_y[j];
                    const double delta_z = zi - points_z[j];
                    const double dist_squared = delta_x * delta_x +
                                                delta_y * delta_y +
                                                delta_z * delta_z;
                    const double inv_dist = 1.0 / sqrt(dist_squared);
                    energy_scalar.add(inv_dist);
                    const double inv_dist_cubed = inv_dist / dist_squared;
                    fx_scalar.add(delta_x * inv_dist_cubed);
                    fy_scalar.add(delta_y * inv_dist_cubed);
                    fz_scalar.add(delta_z * inv_dist_cubed);
                }
                ++j;
            }
        }
        forces_x[i] = fx_vector.to_double() + fx_scalar.to_double();
        forces_y[i] = fy_vector.to_double() + fy_scalar.to_double();
        forces_z[i] = fz_vector.to_double() + fz_scalar.to_double();
    }
    return energy_vector.to_double() + energy_scalar.to_double();
#else
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
#endif // RIESZOLVE_USE_AVX512
}


double constrain_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) noexcept {
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
) noexcept {
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
) noexcept {
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
) noexcept {
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
) noexcept {
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
