#pragma once


class RieszolveOptimizer {


    static constexpr int CHUNK_SIZE = 8;


    double *data;
    double energy;
    double prev_energy;
    double force_norm_squared;
    double prev_force_norm_squared;
    double step_norm_squared;
    double prev_step_norm_squared;
    double last_step_size;
    int num_points;
    int num_chunks;
    int num_iterations;


    constexpr double *subarray(int index) const noexcept {
        return data + index * CHUNK_SIZE * num_chunks;
    }


    constexpr double *points_x() const noexcept { return subarray(0); }
    constexpr double *points_y() const noexcept { return subarray(1); }
    constexpr double *points_z() const noexcept { return subarray(2); }
    constexpr double *forces_x() const noexcept { return subarray(3); }
    constexpr double *forces_y() const noexcept { return subarray(4); }
    constexpr double *forces_z() const noexcept { return subarray(5); }
    constexpr double *temp_points_x() const noexcept { return subarray(6); }
    constexpr double *temp_points_y() const noexcept { return subarray(7); }
    constexpr double *temp_points_z() const noexcept { return subarray(8); }
    constexpr double *prev_forces_x() const noexcept { return subarray(9); }
    constexpr double *prev_forces_y() const noexcept { return subarray(10); }
    constexpr double *prev_forces_z() const noexcept { return subarray(11); }
    constexpr double *step_x() const noexcept { return subarray(12); }
    constexpr double *step_y() const noexcept { return subarray(13); }
    constexpr double *step_z() const noexcept { return subarray(14); }


    void copy_subarray(
        double *__restrict__ dst, const double *__restrict__ src
    ) noexcept;
    void copy_triple_subarray(
        double *__restrict__ dst, const double *__restrict__ src
    ) noexcept;


    void update_forces() noexcept;


public:


    explicit RieszolveOptimizer(int num_points) noexcept;
    ~RieszolveOptimizer() noexcept;
    bool is_allocated() const noexcept;


    double get_energy() const noexcept;
    double get_rms_force() const noexcept;
    double get_last_step_size() const noexcept;
    int get_num_iterations() const noexcept;
    double get_last_step_length() const noexcept;


    void randomize_points(unsigned int seed) noexcept;
    void gradient_descent_step() noexcept;
    void conjugate_gradient_step() noexcept;


    void output_data(
        double *__restrict__ points_x,
        double *__restrict__ points_y,
        double *__restrict__ points_z,
        double *__restrict__ forces_x,
        double *__restrict__ forces_y,
        double *__restrict__ forces_z
    ) noexcept;


}; // class RieszolveOptimizer
