#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "ConvexHull.hpp"
#include "RenderCircle.hpp"
#include "RieszolveOptimizer.hpp"


static inline float rand_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}


static inline SDL_FColor rand_color() {
    const float r = rand_float();
    const float g = rand_float();
    const float b = rand_float();
    const float scale = 1.0f / std::fmaxf(std::fmaxf(r, g), b);
    return {r * scale, g * scale, b * scale, SDL_ALPHA_OPAQUE_FLOAT};
}


// Rieszolve uses mixed-precision arithmetic to solve the Thomson problem
// with extremely high accuracy without sacrificing rendering performance.
// * Geometry optimization is performed in extended precision
//   using compensated algorithms and floating-point expansions.
// * World-space-to-view-space transformations (trigonometry
//   and rotations) are performed in double precision.
// * View-space-to-screen-space transformations (translation
//   and scaling) are performed in single precision.
constexpr double PI = 3.1415926535897932;


namespace GlobalVariables {

constexpr int INITIAL_WINDOW_WIDTH = 1920;
constexpr int INITIAL_WINDOW_HEIGHT = 1080;
constexpr int MAX_NUM_POINTS = 99999;
constexpr int MAX_NUM_FACES = 2 * MAX_NUM_POINTS - 4;

static int num_points = 0;
static int num_faces = 0;
static SDL_Time last_draw_time = 0;
static SDL_Time last_step_time = 0;
static SDL_Time last_draw_duration = 0;
static SDL_Time last_step_duration = 0;
static double angle = 0.0;
static int angular_velocity = 0;
static bool conjugate_gradient = false;
static bool render_forces = false;
static bool render_neighbors = false;
static bool randomize_requested = false;
static bool quit = false;

static RieszolveOptimizer *optimizer = nullptr;
static SDL_Thread *optimizer_thread = nullptr;

static SDL_Window *window = nullptr;
static SDL_Renderer *renderer = nullptr;
static SDL_RWLock *renderer_lock = nullptr;
static double *renderer_points_x = nullptr;
static double *renderer_points_y = nullptr;
static double *renderer_points_z = nullptr;
static double *renderer_forces_x = nullptr;
static double *renderer_forces_y = nullptr;
static double *renderer_forces_z = nullptr;
static int *renderer_faces = nullptr;
static int *renderer_neighbors = nullptr;
static float *screen_points = nullptr;
static float *screen_forces = nullptr;
static SDL_FColor *renderer_colors = nullptr;
static SDL_Vertex *renderer_vertices = nullptr;

} // namespace GlobalVariables


static inline void randomize_colors() {
    using namespace GlobalVariables;
    for (int i = 0; i < num_points; ++i) { renderer_colors[i] = rand_color(); }
}


static inline int SDLCALL run_optimizer(void *) {
    using namespace GlobalVariables;
    while (!quit) {
        if (randomize_requested) {
            const unsigned int seed = static_cast<unsigned int>(time(nullptr));
            optimizer->randomize_points(seed);
            randomize_requested = false;
        }
        // double next_step_size = step_size;
        // const double next_step_norm = compute_step_direction(
        //     optimizer_step_x,
        //     optimizer_step_y,
        //     optimizer_step_z,
        //     optimizer_forces_x,
        //     optimizer_forces_y,
        //     optimizer_forces_z,
        //     optimizer_prev_forces_x,
        //     optimizer_prev_forces_y,
        //     optimizer_prev_forces_z,
        //     num_points,
        //     conjugate_gradient
        // );
        // const double next_energy = quadratic_line_search(
        //     optimizer_points_x,
        //     optimizer_points_y,
        //     optimizer_points_z,
        //     optimizer_temp_x,
        //     optimizer_temp_y,
        //     optimizer_temp_z,
        //     next_step_size,
        //     optimizer_step_x,
        //     optimizer_step_y,
        //     optimizer_step_z,
        //     energy,
        //     num_points
        // );
        // force_norm = update_forces();
        SDL_LockRWLockForWriting(renderer_lock);
        optimizer->output_data(
            renderer_points_x,
            renderer_points_y,
            renderer_points_z,
            renderer_forces_x,
            renderer_forces_y,
            renderer_forces_z
        );
        SDL_UnlockRWLock(renderer_lock);
        // if (next_step_size > 0.0) {
        //     ++num_iterations;
        //     energy = next_energy;
        //     step_size = next_step_size;
        //     step_norm = next_step_norm;
        //     step_length = step_size * step_norm;
        //     SDL_Time current_time;
        //     SDL_GetCurrentTime(&current_time);
        //     last_step_duration = current_time - last_step_time;
        //     last_step_time = current_time;
        // }
    }
    return EXIT_SUCCESS;
}


#define ALLOCATE_MEMORY(POINTER_NAME, ARRAY_TYPE, ARRAY_SIZE)                  \
    do {                                                                       \
        POINTER_NAME = static_cast<ARRAY_TYPE *>(std::malloc(                  \
            static_cast<std::size_t>(ARRAY_SIZE) * sizeof(ARRAY_TYPE)          \
        ));                                                                    \
        if (!(POINTER_NAME)) {                                                 \
            SDL_LogCritical(                                                   \
                SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n"         \
            );                                                                 \
            return SDL_APP_FAILURE;                                            \
        }                                                                      \
    } while (false)


#ifdef _WIN32
#define ALLOCATE_ALIGNED_MEMORY(POINTER_NAME, ARRAY_TYPE, ARRAY_SIZE)          \
    do {                                                                       \
        POINTER_NAME = static_cast<ARRAY_TYPE *>(_aligned_malloc(              \
            static_cast<std::size_t>(ARRAY_SIZE) * sizeof(ARRAY_TYPE), 64      \
        ));                                                                    \
        if (!(POINTER_NAME)) {                                                 \
            SDL_LogCritical(                                                   \
                SDL_LOG_CATEGORY_ERROR, "Failed to allocate aligned memory.\n" \
            );                                                                 \
            return SDL_APP_FAILURE;                                            \
        }                                                                      \
    } while (false)
#else
#define ALLOCATE_ALIGNED_MEMORY(POINTER_NAME, ARRAY_TYPE, ARRAY_SIZE)          \
    do {                                                                       \
        if (posix_memalign(                                                    \
                reinterpret_cast<void **>(&POINTER_NAME),                      \
                64,                                                            \
                static_cast<std::size_t>(ARRAY_SIZE) * sizeof(ARRAY_TYPE)      \
            )) {                                                               \
            SDL_LogCritical(                                                   \
                SDL_LOG_CATEGORY_ERROR, "Failed to allocate aligned memory.\n" \
            );                                                                 \
            return SDL_APP_FAILURE;                                            \
        }                                                                      \
    } while (false)
#endif // _WIN32


#define FREE_MEMORY(POINTER_NAME)                                              \
    do {                                                                       \
        if (POINTER_NAME) { std::free(POINTER_NAME); }                         \
        POINTER_NAME = nullptr;                                                \
    } while (false)


#ifdef _WIN32
#define FREE_ALIGNED_MEMORY(POINTER_NAME)                                      \
    do {                                                                       \
        if (POINTER_NAME) { _aligned_free(POINTER_NAME); }                     \
        POINTER_NAME = nullptr;                                                \
    } while (false)
#else
#define FREE_ALIGNED_MEMORY(POINTER_NAME)                                      \
    do {                                                                       \
        if (POINTER_NAME) { free(POINTER_NAME); }                              \
        POINTER_NAME = nullptr;                                                \
    } while (false)
#endif // _WIN32


static inline int parse_integer(const char *str) {
    using GlobalVariables::MAX_NUM_POINTS;
    char *endptr;
    const long long value = std::strtoll(str, &endptr, 10);
    if (*endptr != '\0') { return -1; }
    if ((value < 4) | (value > static_cast<long long>(MAX_NUM_POINTS))) {
        return -1;
    }
    return static_cast<int>(value);
}


SDL_AppResult SDL_AppInit(void **, int argc, char **argv) {

    using namespace GlobalVariables;

    num_points = 2000;
    for (int i = 1; i < argc; ++i) {
        const int n = parse_integer(argv[i]);
        if (n != -1) { num_points = n; }
    }

    num_faces = 2 * num_points - 4;
    SDL_Time initial_time;
    SDL_GetCurrentTime(&initial_time);
    last_draw_time = initial_time;
    last_step_time = initial_time;
    std::srand(static_cast<unsigned int>(initial_time));

    window = SDL_CreateWindow(
        "Rieszolve",
        INITIAL_WINDOW_WIDTH,
        INITIAL_WINDOW_HEIGHT,
        SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_RESIZABLE
    );
    if (!window) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR,
            "Failed to create window: %s\n",
            SDL_GetError()
        );
        return SDL_APP_FAILURE;
    }

    renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR,
            "Failed to create renderer: %s\n",
            SDL_GetError()
        );
        return SDL_APP_FAILURE;
    }

    renderer_lock = SDL_CreateRWLock();
    if (!renderer_lock) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create lock.\n");
        return SDL_APP_FAILURE;
    }

    ALLOCATE_ALIGNED_MEMORY(renderer_points_x, double, MAX_NUM_POINTS);
    ALLOCATE_ALIGNED_MEMORY(renderer_points_y, double, MAX_NUM_POINTS);
    ALLOCATE_ALIGNED_MEMORY(renderer_points_z, double, MAX_NUM_POINTS);
    ALLOCATE_ALIGNED_MEMORY(renderer_forces_x, double, MAX_NUM_POINTS);
    ALLOCATE_ALIGNED_MEMORY(renderer_forces_y, double, MAX_NUM_POINTS);
    ALLOCATE_ALIGNED_MEMORY(renderer_forces_z, double, MAX_NUM_POINTS);

    ALLOCATE_MEMORY(renderer_faces, int, 3 * MAX_NUM_FACES);
    ALLOCATE_MEMORY(renderer_neighbors, int, MAX_NUM_POINTS);
    ALLOCATE_MEMORY(screen_points, float, 3 * MAX_NUM_POINTS);
    ALLOCATE_MEMORY(screen_forces, float, 2 * MAX_NUM_POINTS);
    ALLOCATE_MEMORY(renderer_colors, SDL_FColor, MAX_NUM_POINTS);
    ALLOCATE_MEMORY(
        renderer_vertices, SDL_Vertex, NUM_CIRCLE_VERTICES * MAX_NUM_POINTS
    );

    optimizer = new (std::nothrow) RieszolveOptimizer(num_points);
    if ((!optimizer) || (!optimizer->is_allocated())) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory for optimizer.\n"
        );
        return SDL_APP_FAILURE;
    }

    const unsigned int seed = static_cast<unsigned int>(time(nullptr));
    optimizer->randomize_points(seed);
    randomize_colors();

    optimizer_thread =
        SDL_CreateThread(run_optimizer, "RieszolveOptimizerThread", nullptr);
    if (!optimizer_thread) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR,
            "Failed to create thread: %s\n",
            SDL_GetError()
        );
        return SDL_APP_FAILURE;
    }

    SDL_Time current_time;
    SDL_GetCurrentTime(&current_time);
    last_draw_duration = current_time - initial_time;
    last_step_duration = current_time - initial_time;

    return SDL_APP_CONTINUE;
}


SDL_AppResult SDL_AppEvent(void *, SDL_Event *event) {
    using namespace GlobalVariables;
    switch (event->type) {
        case SDL_EVENT_QUIT: return SDL_APP_SUCCESS;
        case SDL_EVENT_KEY_DOWN: {
            switch (event->key.key) {
                case SDLK_ESCAPE: return SDL_APP_SUCCESS;
                case SDLK_Q: return SDL_APP_SUCCESS;
                case SDLK_LEFT: angular_velocity -= 2; break;
                case SDLK_RIGHT: angular_velocity += 2; break;
                case SDLK_C: conjugate_gradient = !conjugate_gradient; break;
                case SDLK_F: render_forces = !render_forces; break;
                case SDLK_N:
                    render_neighbors = !render_neighbors;
                    if (!render_neighbors) { randomize_colors(); }
                    break;
                case SDLK_R: randomize_requested = true; break;
                default: break;
            }
            return SDL_APP_CONTINUE;
        }
        default: break;
    }
    return SDL_APP_CONTINUE;
}


constexpr SDL_FColor RED = {1.0f, 0.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor GREEN = {0.0f, 1.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor BLUE = {0.0f, 0.0f, 1.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor YELLOW = {1.0f, 1.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor MAGENTA = {1.0f, 0.0f, 1.0f, SDL_ALPHA_OPAQUE_FLOAT};
constexpr SDL_FColor WHITE = {1.0f, 1.0f, 1.0f, SDL_ALPHA_OPAQUE_FLOAT};


SDL_AppResult SDL_AppIterate(void *) {

    using namespace GlobalVariables;

    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    const float origin_x = 0.5f * static_cast<float>(width);
    const float origin_y = 0.5f * static_cast<float>(height);
    const float scale = 0.375f * static_cast<float>(std::min(width, height));
    const double rms_force = optimizer->get_rms_force();
    const float force_scale = 0.125f * scale / static_cast<float>(rms_force);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    SDL_Time current_time;
    SDL_GetCurrentTime(&current_time);
    last_draw_duration = current_time - last_draw_time;
    last_draw_time = current_time;

    angle += 1.0e-10 * static_cast<double>(angular_velocity) *
             static_cast<double>(last_draw_duration);
    if (angle >= PI) { angle -= (PI + PI); }
    if (angle < -PI) { angle += (PI + PI); }
    const double sin_angle = std::sin(angle);
    const double cos_angle = std::cos(angle);

    SDL_LockRWLockForReading(renderer_lock);
    for (int i = 0; i < num_points; ++i) {
        // Transform world space to view space in double precision.
        const double x = renderer_points_x[i];
        const double y = renderer_points_y[i];
        const double z = renderer_points_z[i];
        const double vx = x * cos_angle + z * sin_angle;
        const double vy = y;
        const double vz = z * cos_angle - x * sin_angle;
        if (!std::signbit(vz)) {
            // Transform view space to screen space in single precision.
            screen_points[3 * i + 0] =
                origin_x + scale * static_cast<float>(vx);
            screen_points[3 * i + 1] =
                origin_y - scale * static_cast<float>(vy);
            // Simulate perspective by making closer points larger.
            // We should use a proper perspective transformation
            // in the future, but this is good enough for now.
            screen_points[3 * i + 2] = 6.0f * static_cast<float>(vz) + 2.0f;
            if (render_forces) {
                const double fx = renderer_forces_x[i];
                const double fy = renderer_forces_y[i];
                const double fz = renderer_forces_z[i];
                const double vfx = fx * cos_angle + fz * sin_angle;
                const double vfy = fy;
                screen_forces[2 * i + 0] =
                    +force_scale * static_cast<float>(vfx);
                screen_forces[2 * i + 1] =
                    -force_scale * static_cast<float>(vfy);
            }
        } else {
            screen_points[3 * i + 0] = NAN;
            screen_points[3 * i + 1] = NAN;
            screen_points[3 * i + 2] = NAN;
            if (render_forces) {
                screen_forces[2 * i + 0] = NAN;
                screen_forces[2 * i + 1] = NAN;
            }
        }
    }
    SDL_UnlockRWLock(renderer_lock);

    const float base_scale = SDL_GetWindowPixelDensity(window);
    SDL_SetRenderScale(renderer, base_scale, base_scale);

    if (render_neighbors) {
        convex_hull(
            renderer_faces,
            renderer_points_x,
            renderer_points_y,
            renderer_points_z,
            num_points
        );
        SDL_SetRenderDrawColor(renderer, 128, 128, 128, SDL_ALPHA_OPAQUE);
        for (int i = 0; i < num_points; ++i) { renderer_neighbors[i] = 0; }
        for (int i = 0; i < num_faces; ++i) {
            const int a_index = renderer_faces[3 * i + 0];
            const int b_index = renderer_faces[3 * i + 1];
            const int c_index = renderer_faces[3 * i + 2];
            if ((a_index >= 0) & (b_index >= 0) & (c_index >= 0)) {
                ++renderer_neighbors[a_index];
                ++renderer_neighbors[b_index];
                ++renderer_neighbors[c_index];
                bool valid = true;
                const float x0 = screen_points[3 * a_index + 0];
                valid &= !std::isnan(x0);
                const float y0 = screen_points[3 * a_index + 1];
                valid &= !std::isnan(y0);
                const float x1 = screen_points[3 * b_index + 0];
                valid &= !std::isnan(x1);
                const float y1 = screen_points[3 * b_index + 1];
                valid &= !std::isnan(y1);
                const float x2 = screen_points[3 * c_index + 0];
                valid &= !std::isnan(x2);
                const float y2 = screen_points[3 * c_index + 1];
                valid &= !std::isnan(y2);
                if (valid) {
                    SDL_RenderLine(renderer, x0, y0, x1, y1);
                    SDL_RenderLine(renderer, x1, y1, x2, y2);
                    SDL_RenderLine(renderer, x2, y2, x0, y0);
                }
            }
        }
        for (int i = 0; i < num_points; ++i) {
            switch (renderer_neighbors[i]) {
                case 4: renderer_colors[i] = YELLOW; break;
                case 5: renderer_colors[i] = RED; break;
                case 6: renderer_colors[i] = GREEN; break;
                case 7: renderer_colors[i] = BLUE; break;
                case 8: renderer_colors[i] = MAGENTA; break;
                default: renderer_colors[i] = WHITE; break;
            }
        }
    }

    int num_rendered_points = 0;
    SDL_Vertex *vertex_pointer = renderer_vertices;
    for (int i = 0; i < num_points; ++i) {
        bool valid = true;
        const float sx = screen_points[3 * i + 0];
        valid &= !std::isnan(sx);
        const float sy = screen_points[3 * i + 1];
        valid &= !std::isnan(sy);
        const float r = screen_points[3 * i + 2];
        valid &= !std::isnan(r);
        if (valid) {
            ++num_rendered_points;
            construct_circle_vertices(
                vertex_pointer, sx, sy, r, renderer_colors[i]
            );
            vertex_pointer += NUM_CIRCLE_VERTICES;
        }
    }

    SDL_RenderGeometry(
        renderer,
        nullptr,
        renderer_vertices,
        NUM_CIRCLE_VERTICES * num_rendered_points,
        nullptr,
        0
    );

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
    if (render_forces) {
        for (int i = 0; i < num_points; ++i) {
            bool valid = true;
            const float sx = screen_points[3 * i + 0];
            valid &= !std::isnan(sx);
            const float sy = screen_points[3 * i + 1];
            valid &= !std::isnan(sy);
            if (valid) {
                SDL_RenderLine(
                    renderer,
                    sx,
                    sy,
                    sx + screen_forces[2 * i + 0],
                    sy + screen_forces[2 * i + 1]
                );
            }
        }
    }

    char debug_message_buffer[256];
    SDL_SetRenderScale(renderer, 2.0f * base_scale, 2.0f * base_scale);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "FPS:%5.0f",
        1.0e9 / static_cast<double>(last_draw_duration)
    );
    SDL_RenderDebugText(renderer, 0.0f, 0.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Angular velocity: %+.1f rad/s",
        0.1 * static_cast<double>(angular_velocity)
    );
    SDL_RenderDebugText(renderer, 0.0f, 10.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Points drawn:%6d",
        num_rendered_points
    );
    SDL_RenderDebugText(renderer, 0.0f, 20.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Iteration count:%8d",
        optimizer->get_num_iterations()
    );
    SDL_RenderDebugText(renderer, 0.0f, 30.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Draw time:%9.3f ms",
        1.0e-6 * static_cast<double>(last_draw_duration)
    );
    SDL_RenderDebugText(renderer, 0.0f, 40.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Step time:%9.3f ms",
        1.0e-6 * static_cast<double>(last_step_duration)
    );
    SDL_RenderDebugText(renderer, 0.0f, 50.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Coulomb energy: %.15e",
        optimizer->get_energy()
    );
    SDL_RenderDebugText(renderer, 0.0f, 60.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "RMS force:      %.15e",
        rms_force
    );
    SDL_RenderDebugText(renderer, 0.0f, 70.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Step size:      %.15e",
        optimizer->get_last_step_size()
    );
    SDL_RenderDebugText(renderer, 0.0f, 80.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Step length:    %.15e",
        optimizer->get_last_step_length()
    );
    SDL_RenderDebugText(renderer, 0.0f, 90.0f, debug_message_buffer);

    SDL_RenderPresent(renderer);

    return SDL_APP_CONTINUE;
}


void SDL_AppQuit(void *, SDL_AppResult) {
    using namespace GlobalVariables;
    quit = true;
    if (optimizer_thread) { SDL_WaitThread(optimizer_thread, nullptr); }
    FREE_MEMORY(renderer_vertices);
    FREE_MEMORY(renderer_colors);
    FREE_MEMORY(screen_forces);
    FREE_MEMORY(screen_points);
    FREE_MEMORY(renderer_neighbors);
    FREE_MEMORY(renderer_faces);
    FREE_ALIGNED_MEMORY(renderer_forces_z);
    FREE_ALIGNED_MEMORY(renderer_forces_y);
    FREE_ALIGNED_MEMORY(renderer_forces_x);
    FREE_ALIGNED_MEMORY(renderer_points_z);
    FREE_ALIGNED_MEMORY(renderer_points_y);
    FREE_ALIGNED_MEMORY(renderer_points_x);
    if (renderer_lock) { SDL_DestroyRWLock(renderer_lock); }
    if (renderer) { SDL_DestroyRenderer(renderer); }
    if (window) { SDL_DestroyWindow(window); }
}
