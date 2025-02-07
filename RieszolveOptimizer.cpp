#include "RieszolveOptimizer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "CoulombEnergy.hpp"


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


void RieszolveOptimizer::update_forces() noexcept {
    copy_triple_subarray(prev_forces_x(), forces_x());
    prev_force_norm_squared = force_norm_squared;
    compute_coulomb_forces(
        forces_x(),
        forces_y(),
        forces_z(),
        points_x(),
        points_y(),
        points_z(),
        num_points
    );
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


RieszolveOptimizer::RieszolveOptimizer(int num_points_arg) noexcept {
    energy = std::nan("");
    prev_energy = std::nan("");
    force_norm_squared = std::nan("");
    prev_force_norm_squared = std::nan("");
    step_norm_squared = std::nan("");
    prev_step_norm_squared = std::nan("");
    last_step_size = std::nan("");
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
    return (data != nullptr);
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
    return last_step_size * std::sqrt(prev_step_norm_squared);
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
    energy = compute_coulomb_energy(px, py, pz, num_points);
    prev_energy = energy;
    update_forces();
    prev_force_norm_squared = force_norm_squared;
    step_norm_squared = force_norm_squared;
    prev_step_norm_squared = force_norm_squared;
    last_step_size = 0.0;
    num_iterations = 0;
    copy_triple_subarray(prev_forces_x(), forces_x());
    copy_triple_subarray(step_x(), forces_x());
}


void RieszolveOptimizer::gradient_descent_step() noexcept {}


void RieszolveOptimizer::conjugate_gradient_step() noexcept {}


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
