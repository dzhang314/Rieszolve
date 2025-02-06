#include "CoulombEnergy.hpp"

#include <cmath>
using std::fma;
using std::sqrt;

#include <cstdlib>
using std::size_t;

#include <cstring>
using std::memcpy;

#ifdef __AVX512F__
#define RIESZOLVE_USE_AVX512
#endif // __AVX512F__

#ifdef RIESZOLVE_USE_AVX512
#include <immintrin.h>
#endif // RIESZOLVE_USE_AVX512


static inline void two_sum(double &s, double &e, double x, double y) {
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


static inline void two_sum(double &s, double &e) {
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

    HighPrecisionAccumulator &operator+=(const HighPrecisionAccumulator &other
    ) noexcept {
        add(other.terms[0]);
        add(other.terms[1]);
        return *this;
    }

}; // struct HighPrecisionAccumulator


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
#endif // RIESZOLVE_USE_AVX512


double compute_coulomb_energy(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
#ifdef RIESZOLVE_USE_AVX512
    const __m512d ONE_VECTOR = _mm512_set1_pd(1.0);
    HighPrecisionVectorAccumulator energy_vector;
    HighPrecisionAccumulator energy_scalar;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : energy_vector, energy_scalar)
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
#pragma omp parallel for schedule(static) reduction(+ : energy)
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
                energy.add(1.0 / sqrt(dist_squared));
            }
        }
    }
    return energy.to_double();
#endif // RIESZOLVE_USE_AVX512
}


void compute_coulomb_forces(
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
#ifdef RIESZOLVE_USE_AVX512
    const __m512d ONE_VECTOR = _mm512_set1_pd(1.0);
#pragma omp parallel for schedule(static)
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
#else
#pragma omp parallel for schedule(static)
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
) {
    HighPrecisionAccumulator force_norm_squared;
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
    return sqrt(force_norm_squared.to_double());
}


static inline double dot_product(
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
) {
    const double force_norm_squared = dot_product(
        forces_x, forces_y, forces_z, forces_x, forces_y, forces_z, num_points
    );
    if (conjugate_gradient) {
        const double prev_norm_squared = dot_product(
            prev_forces_x,
            prev_forces_y,
            prev_forces_z,
            prev_forces_x,
            prev_forces_y,
            prev_forces_z,
            num_points
        );
        const double overlap = dot_product(
            forces_x,
            forces_y,
            forces_z,
            prev_forces_x,
            prev_forces_y,
            prev_forces_z,
            num_points
        );
        if (force_norm_squared > overlap) {
            const double beta =
                (force_norm_squared - overlap) / prev_norm_squared;
#pragma omp simd simdlen(8) aligned(step_x, forces_x : 64)
            for (int i = 0; i < num_points; ++i) {
                step_x[i] = fma(beta, step_x[i], forces_x[i]);
            }
#pragma omp simd simdlen(8) aligned(step_y, forces_y : 64)
            for (int i = 0; i < num_points; ++i) {
                step_y[i] = fma(beta, step_y[i], forces_y[i]);
            }
#pragma omp simd simdlen(8) aligned(step_z, forces_z : 64)
            for (int i = 0; i < num_points; ++i) {
                step_z[i] = fma(beta, step_z[i], forces_z[i]);
            }
            return sqrt(dot_product(
                step_x, step_y, step_z, step_x, step_y, step_z, num_points
            ));
        }
    }
    const size_t size = static_cast<size_t>(num_points) * sizeof(double);
    memcpy(step_x, forces_x, size);
    memcpy(step_y, forces_y, size);
    memcpy(step_z, forces_z, size);
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


static inline double evaluate_step(
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
    move_points(
        new_points_x,
        new_points_y,
        new_points_z,
        points_x,
        points_y,
        points_z,
        step_x,
        step_y,
        step_z,
        step_size,
        num_points
    );
    return compute_coulomb_energy(
        new_points_x, new_points_y, new_points_z, num_points
    );
}


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
) {
    if (!(step_size > 0.0)) {
        step_size = 0.0;
        return initial_energy;
    }
    const size_t size = static_cast<size_t>(num_points) * sizeof(double);
    const double trial_energy = evaluate_step(
        temp_points_x,
        temp_points_y,
        temp_points_z,
        points_x,
        points_y,
        points_z,
        step_x,
        step_y,
        step_z,
        step_size,
        num_points
    );
    if (trial_energy <= initial_energy) { // f1 <= f0
        const double f0 = initial_energy;
        double x1 = step_size;
        double f1 = trial_energy;
        while (true) {
            const double x2 = x1 + x1;
            const double f2 = evaluate_step(
                temp_points_x,
                temp_points_y,
                temp_points_z,
                points_x,
                points_y,
                points_z,
                step_x,
                step_y,
                step_z,
                x2,
                num_points
            );
            if (f2 > f1) {
                const double delta_0 = f0 - f1;
                const double delta_1 = f2 - f1;
                const double delta_sum = delta_0 + delta_1;
                const double numerator = (delta_0 + delta_0) + delta_sum;
                const double denominator = delta_sum + delta_sum;
                const double xq = x1 * (numerator / denominator);
                const double fq = evaluate_step(
                    temp_points_x,
                    temp_points_y,
                    temp_points_z,
                    points_x,
                    points_y,
                    points_z,
                    step_x,
                    step_y,
                    step_z,
                    xq,
                    num_points
                );
                if (fq < f1) {
                    step_size = xq;
                    memcpy(points_x, temp_points_x, size);
                    memcpy(points_y, temp_points_y, size);
                    memcpy(points_z, temp_points_z, size);
                    return fq;
                } else if (f1 < f0) {
                    step_size = x1;
                    move_points(
                        points_x,
                        points_y,
                        points_z,
                        step_x,
                        step_y,
                        step_z,
                        step_size,
                        num_points
                    );
                    return f1;
                } else {
                    step_size = 0.0;
                    return f0;
                }
            }
            x1 = x2;
            f1 = f2;
        }
    } else { // f2 > f0 (or something is NaN)
        const double f0 = initial_energy;
        double x2 = step_size;
        double f2 = trial_energy;
        while (true) {
            const double x1 = 0.5 * x2;
            const double f1 = evaluate_step(
                temp_points_x,
                temp_points_y,
                temp_points_z,
                points_x,
                points_y,
                points_z,
                step_x,
                step_y,
                step_z,
                x1,
                num_points
            );
            if (f1 <= f0) {
                const double delta_0 = f0 - f1;
                const double delta_1 = f2 - f1;
                const double delta_sum = delta_0 + delta_1;
                const double numerator = (delta_0 + delta_0) + delta_sum;
                const double denominator = delta_sum + delta_sum;
                const double xq = x1 * (numerator / denominator);
                const double fq = evaluate_step(
                    temp_points_x,
                    temp_points_y,
                    temp_points_z,
                    points_x,
                    points_y,
                    points_z,
                    step_x,
                    step_y,
                    step_z,
                    xq,
                    num_points
                );
                if (fq < f1) {
                    step_size = xq;
                    memcpy(points_x, temp_points_x, size);
                    memcpy(points_y, temp_points_y, size);
                    memcpy(points_z, temp_points_z, size);
                    return fq;
                } else if (f1 < f0) {
                    step_size = x1;
                    move_points(
                        points_x,
                        points_y,
                        points_z,
                        step_x,
                        step_y,
                        step_z,
                        step_size,
                        num_points
                    );
                    return f1;
                } else {
                    step_size = 0.0;
                    return f0;
                }
            }
            x2 = x1;
            f2 = f1;
        }
    }
}
