#pragma once


class RieszolveOptimizer {

    static constexpr int CHUNK_SIZE = 8;

    double *data;
    double energy;
    double force_norm_squared;
    double prev_force_norm_squared;
    double last_step_size;
    double last_step_length_squared;
    int num_points;
    int num_chunks;
    int num_iterations;

    double *subarray(int index) const noexcept {
        return data + index * CHUNK_SIZE * num_chunks;
    }

    double *points_x() const noexcept { return subarray(0); }
    double *points_y() const noexcept { return subarray(1); }
    double *points_z() const noexcept { return subarray(2); }
    double *forces_x() const noexcept { return subarray(3); }
    double *forces_y() const noexcept { return subarray(4); }
    double *forces_z() const noexcept { return subarray(5); }
    double *temp_points_x() const noexcept { return subarray(6); }
    double *temp_points_y() const noexcept { return subarray(7); }
    double *temp_points_z() const noexcept { return subarray(8); }
    double *prev_forces_x() const noexcept { return subarray(9); }
    double *prev_forces_y() const noexcept { return subarray(10); }
    double *prev_forces_z() const noexcept { return subarray(11); }
    double *step_x() const noexcept { return subarray(12); }
    double *step_y() const noexcept { return subarray(13); }
    double *step_z() const noexcept { return subarray(14); }

    void copy_subarray(
        double *__restrict__ dst, const double *__restrict__ src
    ) noexcept;
    void copy_triple_subarray(
        double *__restrict__ dst, const double *__restrict__ src
    ) noexcept;

    void compute_energy_and_forces() noexcept;
    double move_temp_points(double step_size) noexcept;
    bool quadratic_line_search_helper(
        double x1, double l1, double f0, double f1, double f2
    ) noexcept;
    bool quadratic_line_search() noexcept;

public:

    explicit RieszolveOptimizer(int num_points) noexcept;
    RieszolveOptimizer(const RieszolveOptimizer &) = delete;
    RieszolveOptimizer &operator=(const RieszolveOptimizer &) = delete;
    ~RieszolveOptimizer() noexcept;
    bool is_allocated() const noexcept;

    int get_num_iterations() const noexcept;
    double get_energy() const noexcept;
    double get_rms_force() const noexcept;
    double get_last_step_size() const noexcept;
    double get_rms_step_length() const noexcept;

    void randomize_points(unsigned int seed) noexcept;
    bool gradient_descent_step() noexcept;
    bool conjugate_gradient_step() noexcept;

    void export_points(
        double *__restrict__ points_x,
        double *__restrict__ points_y,
        double *__restrict__ points_z
    ) noexcept;
    void export_forces(
        double *__restrict__ forces_x,
        double *__restrict__ forces_y,
        double *__restrict__ forces_z
    ) noexcept;

}; // class RieszolveOptimizer
