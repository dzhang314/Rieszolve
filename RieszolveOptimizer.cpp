#include "RieszolveOptimizer.hpp"

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "RieszKernels.hpp"


RieszolveOptimizer::RieszolveOptimizer(int num_points_arg) noexcept {
    energy = std::nan("");
    force_norm_squared = std::nan("");
    prev_force_norm_squared = std::nan("");
    last_step_size = std::nan("");
    last_step_length = std::nan("");
    num_points = num_points_arg;
    num_chunks = (num_points_arg + (CHUNK_SIZE - 1)) / CHUNK_SIZE;
    num_iterations = 0;
    const std::size_t alignment = CHUNK_SIZE * sizeof(double);
    const std::size_t size =
        15 * CHUNK_SIZE * static_cast<std::size_t>(num_chunks) * sizeof(double);
#ifdef _WIN32
    data = static_cast<double *>(_aligned_malloc(size, alignment));
#else
    posix_memalign(reinterpret_cast<void **>(&data), alignment, size);
#endif // _WIN32
}


RieszolveOptimizer::~RieszolveOptimizer() noexcept {
    if (data) {
#ifdef _WIN32
        _aligned_free(data);
#else
        free(data);
#endif // _WIN32
    }
}


bool RieszolveOptimizer::is_allocated() const noexcept {
    return static_cast<bool>(data);
}


double RieszolveOptimizer::get_energy() const noexcept { return energy; }


double RieszolveOptimizer::get_rms_force() const noexcept {
    return std::sqrt(force_norm_squared / static_cast<double>(num_points));
}


double RieszolveOptimizer::get_last_step_size() const noexcept {
    return last_step_size;
}


int RieszolveOptimizer::get_num_iterations() const noexcept {
    return num_iterations;
}


double RieszolveOptimizer::get_last_step_length() const noexcept {
    return last_step_length;
}


void RieszolveOptimizer::copy_subarray(
    double *__restrict__ dst, const double *__restrict__ src
) noexcept {
    const std::size_t size =
        static_cast<std::size_t>(num_points) * sizeof(double);
    std::memcpy(dst, src, size);
}


void RieszolveOptimizer::copy_triple_subarray(
    double *__restrict__ dst, const double *__restrict__ src
) noexcept {
    const std::size_t size =
        3 * CHUNK_SIZE * static_cast<std::size_t>(num_chunks) * sizeof(double);
    std::memcpy(dst, src, size);
}


void RieszolveOptimizer::compute_energy_and_forces() noexcept {
    copy_triple_subarray(prev_forces_x(), forces_x());
    energy = compute_coulomb_forces(
        forces_x(),
        forces_y(),
        forces_z(),
        points_x(),
        points_y(),
        points_z(),
        num_points
    );
    prev_force_norm_squared = force_norm_squared;
    force_norm_squared = constrain_forces(
        forces_x(),
        forces_y(),
        forces_z(),
        points_x(),
        points_y(),
        points_z(),
        num_points
    );
}


static inline float rand_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}


void RieszolveOptimizer::randomize_points(unsigned int seed) noexcept {
    double *const px = points_x();
    double *const py = points_y();
    double *const pz = points_z();
    std::srand(seed);
    for (int i = 0; i < num_points; ++i) {
        double x, y, z;
        while (true) {
            x = 2.0 * static_cast<double>(rand_float()) - 1.0;
            y = 2.0 * static_cast<double>(rand_float()) - 1.0;
            z = 2.0 * static_cast<double>(rand_float()) - 1.0;
            const double norm_squared = x * x + y * y + z * z;
            if (norm_squared <= 1.0) {
                const double inv_norm = 1.0 / std::sqrt(norm_squared);
                x *= inv_norm;
                y *= inv_norm;
                z *= inv_norm;
                break;
            }
        }
        px[i] = x;
        py[i] = y;
        pz[i] = z;
    }
    compute_energy_and_forces();
    prev_force_norm_squared = force_norm_squared;
    last_step_size = 0.0;
    last_step_length = 0.0;
    num_iterations = 0;
    copy_triple_subarray(prev_forces_x(), forces_x());
    copy_triple_subarray(step_x(), forces_x());
}


double RieszolveOptimizer::move_temp_points(double step_size) noexcept {
    return move_points(
        temp_points_x(),
        temp_points_y(),
        temp_points_z(),
        points_x(),
        points_y(),
        points_z(),
        step_x(),
        step_y(),
        step_z(),
        step_size,
        num_points
    );
}


bool RieszolveOptimizer::quadratic_line_search_helper(
    double x1, double l1, double f0, double f1, double f2
) noexcept {
    const double delta_0 = f0 - f1;
    const double delta_1 = f2 - f1;
    const double delta_sum = delta_0 + delta_1;
    const double numerator = (delta_0 + delta_0) + delta_sum;
    const double denominator = delta_sum + delta_sum;
    const double xq = x1 * (numerator / denominator);
    const double lq = std::sqrt(move_temp_points(xq));
    const double fq = compute_coulomb_energy(
        temp_points_x(), temp_points_y(), temp_points_z(), num_points
    );
    if (fq <= f1) { // return (xq, fq)
        last_step_size = xq;
        last_step_length = lq;
        energy = fq;
        copy_triple_subarray(points_x(), temp_points_x());
        ++num_iterations;
        return true;
    } else if (f1 < f0) { // return (x1, f1)
        last_step_size = x1;
        last_step_length = l1;
        energy = f1;
        move_points(
            points_x(),
            points_y(),
            points_z(),
            step_x(),
            step_y(),
            step_z(),
            x1,
            num_points
        );
        ++num_iterations;
        return true;
    } else { // return (0, f0)
        return false;
    }
}


bool RieszolveOptimizer::quadratic_line_search() noexcept {
    const double f0 = energy;
    double initial_step_size =
        (last_step_size > 0.0) ? last_step_size : DBL_EPSILON;
    double initial_step_length = std::sqrt(move_temp_points(initial_step_size));
    while (initial_step_length == 0.0) {
        initial_step_size += initial_step_size;
        initial_step_length = std::sqrt(move_temp_points(initial_step_size));
    }
    const double initial_step_energy = compute_coulomb_energy(
        temp_points_x(), temp_points_y(), temp_points_z(), num_points
    );
    if (!std::isfinite(initial_step_energy)) { // return (0, f0)
        return false;
    } else if (initial_step_energy <= f0) { // increase step size
        const double x1 = initial_step_size;
        const double l1 = initial_step_length;
        const double f1 = initial_step_energy;
        const double x2 = x1 + x1;
        const double l2 = std::sqrt(move_temp_points(x2));
        const double f2 = compute_coulomb_energy(
            temp_points_x(), temp_points_y(), temp_points_z(), num_points
        );
        if (!std::isfinite(f2)) { // return (x1, f1)
            last_step_size = x1;
            last_step_length = l1;
            energy = f1;
            move_points(
                points_x(),
                points_y(),
                points_z(),
                step_x(),
                step_y(),
                step_z(),
                x1,
                num_points
            );
            ++num_iterations;
            return true;
        } else if (f2 > f1) { // perform quadratic interpolation
            return quadratic_line_search_helper(x1, l1, f0, f1, f2);
        } else { // return (x2, f2)
            last_step_size = x2;
            last_step_length = l2;
            energy = f2;
            copy_triple_subarray(points_x(), temp_points_x());
            ++num_iterations;
            return true;
        }
    } else { // decrease step size
        double x2 = initial_step_size;
        double l2 = initial_step_length;
        double f2 = initial_step_energy;
        while (true) {
            const double x1 = 0.5 * x2;
            const double l1 = std::sqrt(move_temp_points(x1));
            if (!(l1 < l2)) { // return (0, f0)
                return false;
            }
            const double f1 = compute_coulomb_energy(
                temp_points_x(), temp_points_y(), temp_points_z(), num_points
            );
            if (f1 <= f0) { // perform quadratic interpolation
                return quadratic_line_search_helper(x1, l1, f0, f1, f2);
            }
            x2 = x1;
            l2 = l1;
            f2 = f1;
        }
    }
}


bool RieszolveOptimizer::gradient_descent_step() noexcept {
    copy_triple_subarray(step_x(), forces_x());
    const bool success = quadratic_line_search();
    if (success) { compute_energy_and_forces(); }
    return success;
}


bool RieszolveOptimizer::conjugate_gradient_step() noexcept {
    const double overlap = dot_product(
        forces_x(),
        forces_y(),
        forces_z(),
        prev_forces_x(),
        prev_forces_y(),
        prev_forces_z(),
        num_points
    );
    if (force_norm_squared > overlap) {
        const double beta =
            (force_norm_squared - overlap) / prev_force_norm_squared;
        xpay(
            step_x(),
            step_y(),
            step_z(),
            beta,
            forces_x(),
            forces_y(),
            forces_z(),
            num_points
        );
    } else {
        copy_triple_subarray(step_x(), forces_x());
    }
    const bool success = quadratic_line_search();
    if (success) { compute_energy_and_forces(); }
    return success;
}


void RieszolveOptimizer::output_data(
    double *__restrict__ points_x_arg,
    double *__restrict__ points_y_arg,
    double *__restrict__ points_z_arg,
    double *__restrict__ forces_x_arg,
    double *__restrict__ forces_y_arg,
    double *__restrict__ forces_z_arg
) noexcept {
    copy_subarray(points_x_arg, points_x());
    copy_subarray(points_y_arg, points_y());
    copy_subarray(points_z_arg, points_z());
    copy_subarray(forces_x_arg, forces_x());
    copy_subarray(forces_y_arg, forces_y());
    copy_subarray(forces_z_arg, forces_z());
}
