#include "RieszolveRenderer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

#include <SDL3/SDL.h>


constexpr int NUM_CIRCLE_VERTICES = 36;


RieszolveRenderer::RieszolveRenderer(int num_points_) noexcept {
    num_points = num_points_;
    if (num_points < 0) { num_points = 0; }
    int num_faces = 2 * num_points - 4;
    if (num_faces < 0) { num_faces = 0; }
    double_data = new (std::nothrow) double[6 * num_points];
    float_data = new (std::nothrow) float[5 * num_points];
    integer_data = new (std::nothrow) int[num_points + 3 * num_faces];
    color_data = new (std::nothrow) SDL_FColor[num_points];
    vertex_data =
        new (std::nothrow) SDL_Vertex[NUM_CIRCLE_VERTICES * num_points];
}


RieszolveRenderer::~RieszolveRenderer() noexcept {
    if (vertex_data) { delete[] vertex_data; }
    if (color_data) { delete[] color_data; }
    if (integer_data) { delete[] integer_data; }
    if (float_data) { delete[] float_data; }
    if (double_data) { delete[] double_data; }
}


bool RieszolveRenderer::is_allocated() const noexcept {
    return (
        static_cast<bool>(double_data) & static_cast<bool>(float_data) &
        static_cast<bool>(integer_data) & static_cast<bool>(color_data) &
        static_cast<bool>(vertex_data)
    );
}


void RieszolveRenderer::import_points(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z
) noexcept {
    const std::size_t size =
        static_cast<std::size_t>(num_points) * sizeof(double);
    std::memcpy(double_data + 0 * num_points, points_x, size);
    std::memcpy(double_data + 1 * num_points, points_y, size);
    std::memcpy(double_data + 2 * num_points, points_z, size);
}


void RieszolveRenderer::import_forces(
    const double *__restrict__ forces_x,
    const double *__restrict__ forces_y,
    const double *__restrict__ forces_z
) noexcept {
    const std::size_t size =
        static_cast<std::size_t>(num_points) * sizeof(double);
    std::memcpy(double_data + 3 * num_points, forces_x, size);
    std::memcpy(double_data + 4 * num_points, forces_y, size);
    std::memcpy(double_data + 5 * num_points, forces_z, size);
}


void RieszolveRenderer::import_faces(const int *__restrict__ faces) noexcept {
    int num_faces = 2 * num_points - 4;
    if (num_faces < 0) { num_faces = 0; }
    const std::size_t size =
        3 * static_cast<std::size_t>(num_faces) * sizeof(int);
    std::memcpy(integer_data + num_points, faces, size);
}


static inline float rand_float() noexcept {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}


static inline SDL_FColor rand_color() noexcept {
    const float r = rand_float();
    const float g = rand_float();
    const float b = rand_float();
    const float scale = 1.0f / std::fmaxf(std::fmaxf(r, g), b);
    return {r * scale, g * scale, b * scale, SDL_ALPHA_OPAQUE_FLOAT};
}


void RieszolveRenderer::randomize_colors() noexcept {
    for (int i = 0; i < num_points; ++i) { color_data[i] = rand_color(); }
}


constexpr SDL_FColor RED = {1.0f, 0.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor GREEN = {0.0f, 1.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor BLUE = {0.0f, 0.0f, 1.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor CYAN = {0.0f, 1.0f, 1.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor MAGENTA = {1.0f, 0.0f, 1.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor YELLOW = {1.0f, 1.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor WHITE = {1.0f, 1.0f, 1.0f, SDL_ALPHA_OPAQUE_FLOAT};


void RieszolveRenderer::compute_nearest_neighbor_colors() noexcept {
    for (int i = 0; i < num_points; ++i) { integer_data[i] = 0; }
    const int num_faces = 2 * num_points - 4;
    for (int i = 0; i < 3 * num_faces; ++i) {
        const int index = integer_data[num_points + i];
        if (index >= 0) { ++integer_data[index]; }
    }
    for (int i = 0; i < num_points; ++i) {
        switch (integer_data[i]) {
            case 3: color_data[i] = CYAN; break;
            case 4: color_data[i] = YELLOW; break;
            case 5: color_data[i] = RED; break;
            case 6: color_data[i] = GREEN; break;
            case 7: color_data[i] = BLUE; break;
            case 8: color_data[i] = MAGENTA; break;
            default: color_data[i] = WHITE; break;
        }
    }
}


void RieszolveRenderer::compute_screen_points(
    double angle, float origin_x, float origin_y, float scale
) noexcept {
    const double cos_angle = std::cos(angle);
    const double sin_angle = std::sin(angle);
    for (int i = 0; i < num_points; ++i) {
        const double x = double_data[0 * num_points + i];
        const double y = double_data[1 * num_points + i];
        const double z = double_data[2 * num_points + i];
        const double vx = x * cos_angle + z * sin_angle;
        const double vy = y;
        const double vz = z * cos_angle - x * sin_angle;
        if (std::signbit(vz)) {
            float_data[3 * i + 0] = NAN;
            float_data[3 * i + 1] = NAN;
            float_data[3 * i + 2] = NAN;
        } else {
            float_data[3 * i + 0] = origin_x + scale * static_cast<float>(vx);
            float_data[3 * i + 1] = origin_y - scale * static_cast<float>(vy);
            float_data[3 * i + 2] = 6.0f * static_cast<float>(vz) + 2.0f;
        }
    }
}


void RieszolveRenderer::compute_screen_forces(
    double angle, float scale
) noexcept {
    const double cos_angle = std::cos(angle);
    const double sin_angle = std::sin(angle);
    for (int i = 0; i < num_points; ++i) {
        const double fx = double_data[3 * num_points + i];
        const double fy = double_data[4 * num_points + i];
        const double fz = double_data[5 * num_points + i];
        const double vx = fx * cos_angle + fz * sin_angle;
        const double vy = fy;
        float_data[3 * num_points + 2 * i + 0] =
            +scale * static_cast<float>(vx);
        float_data[3 * num_points + 2 * i + 1] =
            -scale * static_cast<float>(vy);
    }
}


void RieszolveRenderer::render_nearest_neighbor_mesh(SDL_Renderer *renderer
) noexcept {
    int num_faces = 2 * num_points - 4;
    if (num_faces < 0) { num_faces = 0; }
    for (int i = 0; i < num_faces; ++i) {
        const int a_index = integer_data[num_points + 3 * i + 0];
        const int b_index = integer_data[num_points + 3 * i + 1];
        const int c_index = integer_data[num_points + 3 * i + 2];
        if ((a_index >= 0) & (b_index >= 0) & (c_index >= 0)) {
            bool valid = true;
            const float x0 = float_data[3 * a_index + 0];
            valid &= !std::isnan(x0);
            const float y0 = float_data[3 * a_index + 1];
            valid &= !std::isnan(y0);
            const float x1 = float_data[3 * b_index + 0];
            valid &= !std::isnan(x1);
            const float y1 = float_data[3 * b_index + 1];
            valid &= !std::isnan(y1);
            const float x2 = float_data[3 * c_index + 0];
            valid &= !std::isnan(x2);
            const float y2 = float_data[3 * c_index + 1];
            valid &= !std::isnan(y2);
            if (valid) {
                SDL_RenderLine(renderer, x0, y0, x1, y1);
                SDL_RenderLine(renderer, x1, y1, x2, y2);
                SDL_RenderLine(renderer, x2, y2, x0, y0);
            }
        }
    }
}


static inline void construct_circle_vertices(
    SDL_Vertex *vertices, float x, float y, float radius, SDL_FColor color
) noexcept {
    const float major = 0.866025404f * radius;
    const float minor = 0.5f * radius;

    vertices[0].position.x = x;
    vertices[0].position.y = y;
    vertices[0].color = color;
    vertices[1].position.x = x + radius;
    vertices[1].position.y = y;
    vertices[1].color = color;
    vertices[2].position.x = x + major;
    vertices[2].position.y = y + minor;
    vertices[2].color = color;

    vertices[3].position.x = x;
    vertices[3].position.y = y;
    vertices[3].color = color;
    vertices[4].position.x = x + major;
    vertices[4].position.y = y + minor;
    vertices[4].color = color;
    vertices[5].position.x = x + minor;
    vertices[5].position.y = y + major;
    vertices[5].color = color;

    vertices[6].position.x = x;
    vertices[6].position.y = y;
    vertices[6].color = color;
    vertices[7].position.x = x + minor;
    vertices[7].position.y = y + major;
    vertices[7].color = color;
    vertices[8].position.x = x;
    vertices[8].position.y = y + radius;
    vertices[8].color = color;

    vertices[9].position.x = x;
    vertices[9].position.y = y;
    vertices[9].color = color;
    vertices[10].position.x = x;
    vertices[10].position.y = y + radius;
    vertices[10].color = color;
    vertices[11].position.x = x - minor;
    vertices[11].position.y = y + major;
    vertices[11].color = color;

    vertices[12].position.x = x;
    vertices[12].position.y = y;
    vertices[12].color = color;
    vertices[13].position.x = x - minor;
    vertices[13].position.y = y + major;
    vertices[13].color = color;
    vertices[14].position.x = x - major;
    vertices[14].position.y = y + minor;
    vertices[14].color = color;

    vertices[15].position.x = x;
    vertices[15].position.y = y;
    vertices[15].color = color;
    vertices[16].position.x = x - major;
    vertices[16].position.y = y + minor;
    vertices[16].color = color;
    vertices[17].position.x = x - radius;
    vertices[17].position.y = y;
    vertices[17].color = color;

    vertices[18].position.x = x;
    vertices[18].position.y = y;
    vertices[18].color = color;
    vertices[19].position.x = x - radius;
    vertices[19].position.y = y;
    vertices[19].color = color;
    vertices[20].position.x = x - major;
    vertices[20].position.y = y - minor;
    vertices[20].color = color;

    vertices[21].position.x = x;
    vertices[21].position.y = y;
    vertices[21].color = color;
    vertices[22].position.x = x - major;
    vertices[22].position.y = y - minor;
    vertices[22].color = color;
    vertices[23].position.x = x - minor;
    vertices[23].position.y = y - major;
    vertices[23].color = color;

    vertices[24].position.x = x;
    vertices[24].position.y = y;
    vertices[24].color = color;
    vertices[25].position.x = x - minor;
    vertices[25].position.y = y - major;
    vertices[25].color = color;
    vertices[26].position.x = x;
    vertices[26].position.y = y - radius;
    vertices[26].color = color;

    vertices[27].position.x = x;
    vertices[27].position.y = y;
    vertices[27].color = color;
    vertices[28].position.x = x;
    vertices[28].position.y = y - radius;
    vertices[28].color = color;
    vertices[29].position.x = x + minor;
    vertices[29].position.y = y - major;
    vertices[29].color = color;

    vertices[30].position.x = x;
    vertices[30].position.y = y;
    vertices[30].color = color;
    vertices[31].position.x = x + minor;
    vertices[31].position.y = y - major;
    vertices[31].color = color;
    vertices[32].position.x = x + major;
    vertices[32].position.y = y - minor;
    vertices[32].color = color;

    vertices[33].position.x = x;
    vertices[33].position.y = y;
    vertices[33].color = color;
    vertices[34].position.x = x + major;
    vertices[34].position.y = y - minor;
    vertices[34].color = color;
    vertices[35].position.x = x + radius;
    vertices[35].position.y = y;
    vertices[35].color = color;
}


int RieszolveRenderer::render_points(SDL_Renderer *renderer) noexcept {
    int num_rendered_points = 0;
    for (int i = 0; i < num_points; ++i) {
        bool valid = true;
        const float sx = float_data[3 * i + 0];
        valid &= !std::isnan(sx);
        const float sy = float_data[3 * i + 1];
        valid &= !std::isnan(sy);
        const float r = float_data[3 * i + 2];
        valid &= !std::isnan(r);
        if (valid) {
            construct_circle_vertices(
                vertex_data + num_rendered_points * NUM_CIRCLE_VERTICES,
                sx,
                sy,
                r,
                color_data[i]
            );
            ++num_rendered_points;
        }
    }
    SDL_RenderGeometry(
        renderer,
        nullptr,
        vertex_data,
        NUM_CIRCLE_VERTICES * num_rendered_points,
        nullptr,
        0
    );
    return num_rendered_points;
}


void RieszolveRenderer::render_forces(SDL_Renderer *renderer) noexcept {
    for (int i = 0; i < num_points; ++i) {
        bool valid = true;
        const float sx = float_data[3 * i + 0];
        valid &= !std::isnan(sx);
        const float sy = float_data[3 * i + 1];
        valid &= !std::isnan(sy);
        const float fx = float_data[3 * num_points + 2 * i + 0];
        valid &= !std::isnan(sx);
        const float fy = float_data[3 * num_points + 2 * i + 1];
        valid &= !std::isnan(sy);
        if (valid) { SDL_RenderLine(renderer, sx, sy, sx + fx, sy + fy); }
    }
}
