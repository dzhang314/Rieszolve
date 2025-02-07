#include "RieszolveOptimizer.hpp"

#include <cstdlib>
#include <cstring>


RieszolveOptimizer::RieszolveOptimizer(int num_points) noexcept {
    this->num_points = num_points;
    this->num_chunks = (num_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const std::size_t alignment = CHUNK_SIZE * sizeof(double);
    const std::size_t size =
        15 * CHUNK_SIZE * static_cast<std::size_t>(num_chunks) * sizeof(double);
#ifdef _WIN32
    this->data = static_cast<double *>(_aligned_malloc(size, alignment));
#else
    posix_memalign(reinterpret_cast<void **>(&this->data), alignment, size);
#endif // _WIN32
}


RieszolveOptimizer::~RieszolveOptimizer() noexcept {
    if (this->data) {
#ifdef _WIN32
        _aligned_free(this->data);
#else
        free(this->data);
#endif // _WIN32
    }
}


void RieszolveOptimizer::randomize_points(int seed) noexcept {}


void RieszolveOptimizer::update_forces() noexcept {}


void RieszolveOptimizer::gradient_descent_step() noexcept {}


void RieszolveOptimizer::conjugate_gradient_step() noexcept {}


void RieszolveOptimizer::output_data(
    double *__restrict__ points_x,
    double *__restrict__ points_y,
    double *__restrict__ points_z,
    double *__restrict__ forces_x,
    double *__restrict__ forces_y,
    double *__restrict__ forces_z
) noexcept {}
