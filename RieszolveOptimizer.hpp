#pragma once


class RieszolveOptimizer {


    static constexpr int CHUNK_SIZE = 8;


    double *data;
    int num_points;
    int num_chunks;


    constexpr double *chunk(int index) const noexcept {
        return data + index * CHUNK_SIZE * num_chunks;
    }


    constexpr double *points_x() const noexcept { return chunk(0); }
    constexpr double *points_y() const noexcept { return chunk(1); }
    constexpr double *points_z() const noexcept { return chunk(2); }
    constexpr double *forces_x() const noexcept { return chunk(3); }
    constexpr double *forces_y() const noexcept { return chunk(4); }
    constexpr double *forces_z() const noexcept { return chunk(5); }
    constexpr double *temp_points_x() const noexcept { return chunk(6); }
    constexpr double *temp_points_y() const noexcept { return chunk(7); }
    constexpr double *temp_points_z() const noexcept { return chunk(8); }
    constexpr double *prev_forces_x() const noexcept { return chunk(9); }
    constexpr double *prev_forces_y() const noexcept { return chunk(10); }
    constexpr double *prev_forces_z() const noexcept { return chunk(11); }
    constexpr double *step_x() const noexcept { return chunk(12); }
    constexpr double *step_y() const noexcept { return chunk(13); }
    constexpr double *step_z() const noexcept { return chunk(14); }


public:


    explicit RieszolveOptimizer(int num_points) noexcept;


    ~RieszolveOptimizer() noexcept;


    constexpr bool is_allocated() const noexcept { return (data != nullptr); }


    void randomize_points(int seed);


    void update_forces() noexcept;


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
