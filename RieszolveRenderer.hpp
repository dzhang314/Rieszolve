#pragma once

#include <SDL3/SDL_pixels.h> // for SDL_FColor
#include <SDL3/SDL_render.h> // for SDL_Vertex


class RieszolveRenderer {

    double *__restrict__ double_data;
    // renderer_points_x -- size num_points
    // renderer_points_y -- size num_points
    // renderer_points_z -- size num_points
    // renderer_forces_x -- size num_points
    // renderer_forces_y -- size num_points
    // renderer_forces_z -- size num_points
    float *__restrict__ float_data;
    // screen_points -- size 3 * num_points
    // screen_forces -- size 2 * num_points
    int *__restrict__ integer_data;
    // renderer_neighbors -- size num_points
    // renderer_faces -- size 3 * num_faces
    SDL_FColor *__restrict__ color_data;  // size num_points
    SDL_Vertex *__restrict__ vertex_data; // size NUM_CIRCLE_VERTICES *
                                          // num_points
    int num_points;

public:

    explicit RieszolveRenderer(int num_points) noexcept;
    RieszolveRenderer(const RieszolveRenderer &) = delete;
    RieszolveRenderer &operator=(const RieszolveRenderer &) = delete;
    ~RieszolveRenderer() noexcept;
    bool is_allocated() const noexcept;

    void import_points(
        const double *__restrict__ points_x,
        const double *__restrict__ points_y,
        const double *__restrict__ points_z
    ) noexcept;
    void import_forces(
        const double *__restrict__ forces_x,
        const double *__restrict__ forces_y,
        const double *__restrict__ forces_z
    ) noexcept;
    void import_faces(const int *__restrict__ faces) noexcept;

    void randomize_colors() noexcept;
    void compute_nearest_neighbor_colors() noexcept;
    void compute_screen_points(
        double angle, float origin_x, float origin_y, float scale
    ) noexcept;
    void compute_screen_forces(double angle, float scale) noexcept;

    void render_nearest_neighbor_mesh(SDL_Renderer *renderer) noexcept;
    int render_points(SDL_Renderer *renderer) noexcept;
    void render_forces(SDL_Renderer *renderer) noexcept;

}; // class RieszolveRenderer
