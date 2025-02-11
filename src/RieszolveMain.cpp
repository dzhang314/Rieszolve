#include <algorithm>
#include <atomic>
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
#include "RieszolveOptimizer.hpp"
#include "RieszolveRenderer.hpp"


// Rieszolve uses mixed-precision arithmetic to solve the Thomson problem
// with extremely high accuracy without sacrificing rendering performance.
// * Geometry optimization is performed in extended precision
//   using compensated algorithms and floating-point expansions.
// * World-space-to-view-space transformations (trigonometry and rotations)
//   are performed in double precision.
// * View-space-to-screen-space transformations (translation and scaling)
//   are performed in single precision.
constexpr double PI = 3.1415926535897932;


namespace GlobalVariables {

constexpr int INITIAL_WINDOW_WIDTH = 1280;
constexpr int INITIAL_WINDOW_HEIGHT = 720;
constexpr int MAX_NUM_POINTS = 99999;

static int num_points = 0;
static int num_faces = 0;
static double angle = 0.0;
static int angular_velocity = 0;
static bool render_forces = false;
static bool render_neighbors = false;

static std::atomic<SDL_Time> last_step_time{0};
static std::atomic<SDL_Time> last_hull_time{0};
static std::atomic<SDL_Time> last_draw_time{0};
static std::atomic<SDL_Time> last_step_duration{0};
static std::atomic<SDL_Time> last_hull_duration{0};
static std::atomic<SDL_Time> last_draw_duration{0};
static std::atomic<int> num_iterations{0};
static std::atomic<double> energy{0.0};
static std::atomic<double> rms_force{0.0};
static std::atomic<double> last_step_size{0.0};
static std::atomic<double> rms_step_length{0.0};
static std::atomic<bool> conjugate_gradient{false};
static std::atomic<bool> randomize_requested{false};
static std::atomic<bool> quit{false};

static double *points_x = nullptr;
static double *points_y = nullptr;
static double *points_z = nullptr;
static double *forces_x = nullptr;
static double *forces_y = nullptr;
static double *forces_z = nullptr;
static int *faces = nullptr;

static SDL_Window *sdl_window_handle = nullptr;
static SDL_Renderer *sdl_renderer_handle = nullptr;
static SDL_RWLock *point_force_data_lock = nullptr;
static SDL_RWLock *face_data_lock = nullptr;
static RieszolveOptimizer *optimizer = nullptr;
static RieszolveRenderer *renderer = nullptr;
static SDL_Thread *optimizer_thread = nullptr;
static SDL_Thread *convex_hull_thread = nullptr;

} // namespace GlobalVariables


static inline int SDLCALL run_optimizer(void *) {
    using namespace GlobalVariables;
    while (!quit) {
        if (randomize_requested) {
            SDL_Time seed_time;
            SDL_GetCurrentTime(&seed_time);
            optimizer->randomize_points(static_cast<unsigned>(seed_time));
            randomize_requested = false;
        }
        bool success;
        if (conjugate_gradient) {
            success = optimizer->conjugate_gradient_step();
            if (!success) { success = optimizer->gradient_descent_step(); }
        } else {
            success = optimizer->gradient_descent_step();
        }
        if (success) {
            num_iterations = optimizer->get_num_iterations();
            energy = optimizer->get_energy();
            rms_force = optimizer->get_rms_force();
            last_step_size = optimizer->get_last_step_size();
            rms_step_length = optimizer->get_rms_step_length();
            SDL_LockRWLockForWriting(point_force_data_lock);
            optimizer->export_points(points_x, points_y, points_z);
            optimizer->export_forces(forces_x, forces_y, forces_z);
            SDL_UnlockRWLock(point_force_data_lock);
        }
        SDL_Time current_time;
        SDL_GetCurrentTime(&current_time);
        last_step_duration = current_time - last_step_time;
        last_step_time = current_time;
    }
    return EXIT_SUCCESS;
}


static inline int SDLCALL run_convex_hull(void *) {
    using namespace GlobalVariables;
    const std::size_t point_array_size =
        static_cast<std::size_t>(num_points) * sizeof(double);
    const std::size_t face_array_size =
        3 * static_cast<std::size_t>(num_faces) * sizeof(int);
    double *local_points_x = new double[num_points];
    double *local_points_y = new double[num_points];
    double *local_points_z = new double[num_points];
    int *local_faces = new int[3 * num_faces];
    while (!quit) {
        SDL_LockRWLockForReading(point_force_data_lock);
        std::memcpy(local_points_x, points_x, point_array_size);
        std::memcpy(local_points_y, points_y, point_array_size);
        std::memcpy(local_points_z, points_z, point_array_size);
        SDL_UnlockRWLock(point_force_data_lock);
        convex_hull(
            local_faces,
            local_points_x,
            local_points_y,
            local_points_z,
            num_points
        );
        SDL_LockRWLockForWriting(face_data_lock);
        std::memcpy(faces, local_faces, face_array_size);
        SDL_UnlockRWLock(face_data_lock);
        SDL_Time current_time;
        SDL_GetCurrentTime(&current_time);
        last_hull_duration = current_time - last_hull_time;
        last_hull_time = current_time;
    }
    delete[] local_faces;
    delete[] local_points_z;
    delete[] local_points_y;
    delete[] local_points_x;
    return EXIT_SUCCESS;
}


static inline int parse_integer(const char *str) {
    using GlobalVariables::MAX_NUM_POINTS;
    char *endptr;
    const long long value = std::strtoll(str, &endptr, 10);
    if (*endptr != '\0') { return -1; }
    if ((value < 2) | (value > static_cast<long long>(MAX_NUM_POINTS))) {
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
    if (num_faces < 0) { num_faces = 0; }

    SDL_Time initial_time;
    SDL_GetCurrentTime(&initial_time);
    last_step_time = initial_time;
    last_hull_time = initial_time;
    last_draw_time = initial_time;

    points_x = new (std::nothrow) double[num_points];
    points_y = new (std::nothrow) double[num_points];
    points_z = new (std::nothrow) double[num_points];
    forces_x = new (std::nothrow) double[num_points];
    forces_y = new (std::nothrow) double[num_points];
    forces_z = new (std::nothrow) double[num_points];
    faces = new (std::nothrow) int[3 * num_faces];
    if ((!points_x) | (!points_y) | (!points_z) | (!forces_x) | (!forces_y) |
        (!forces_z) | (!faces)) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }
    for (int i = 0; i < num_faces; ++i) {
        faces[3 * i + 0] = -1;
        faces[3 * i + 1] = -1;
        faces[3 * i + 2] = -1;
    }

    sdl_window_handle = SDL_CreateWindow(
        "Rieszolve",
        INITIAL_WINDOW_WIDTH,
        INITIAL_WINDOW_HEIGHT,
        SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_RESIZABLE
    );
    if (!sdl_window_handle) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR,
            "Failed to create SDL window: %s\n",
            SDL_GetError()
        );
        return SDL_APP_FAILURE;
    }

    sdl_renderer_handle = SDL_CreateRenderer(sdl_window_handle, nullptr);
    if (!sdl_renderer_handle) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR,
            "Failed to create SDL renderer: %s\n",
            SDL_GetError()
        );
        return SDL_APP_FAILURE;
    }

    point_force_data_lock = SDL_CreateRWLock();
    if (!point_force_data_lock) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL lock.\n");
        return SDL_APP_FAILURE;
    }

    face_data_lock = SDL_CreateRWLock();
    if (!face_data_lock) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL lock.\n");
        return SDL_APP_FAILURE;
    }

    optimizer = new (std::nothrow) RieszolveOptimizer(num_points);
    if ((!optimizer) || (!optimizer->is_allocated())) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR, "Failed to construct optimizer.\n"
        );
        return SDL_APP_FAILURE;
    }

    optimizer->randomize_points(static_cast<unsigned>(initial_time));
    optimizer->export_points(points_x, points_y, points_z);
    optimizer->export_forces(forces_x, forces_y, forces_z);

    renderer = new (std::nothrow) RieszolveRenderer(num_points);
    if ((!renderer) || (!renderer->is_allocated())) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR, "Failed to construct renderer.\n"
        );
        return SDL_APP_FAILURE;
    }
    renderer->import_points(points_x, points_y, points_z);
    renderer->import_forces(forces_x, forces_y, forces_z);
    renderer->import_faces(faces);
    renderer->randomize_colors();

    optimizer_thread =
        SDL_CreateThread(run_optimizer, "RieszolveOptimizerThread", nullptr);
    if (!optimizer_thread) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR,
            "Failed to create SDL thread: %s\n",
            SDL_GetError()
        );
        return SDL_APP_FAILURE;
    }
    convex_hull_thread =
        SDL_CreateThread(run_convex_hull, "RieszolveConvexHullThread", nullptr);
    if (!convex_hull_thread) {
        SDL_LogCritical(
            SDL_LOG_CATEGORY_ERROR,
            "Failed to create SDL thread: %s\n",
            SDL_GetError()
        );
        return SDL_APP_FAILURE;
    }

    SDL_Time current_time;
    SDL_GetCurrentTime(&current_time);
    last_step_duration = current_time - initial_time;
    last_hull_duration = current_time - initial_time;
    last_draw_duration = current_time - initial_time;
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
                    if (!render_neighbors) { renderer->randomize_colors(); }
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


SDL_AppResult SDL_AppIterate(void *) {
    using namespace GlobalVariables;

    int width, height;
    SDL_GetWindowSize(sdl_window_handle, &width, &height);
    const float origin_x = 0.5f * static_cast<float>(width);
    const float origin_y = 0.5f * static_cast<float>(height);
    const float scale = 0.375f * static_cast<float>(std::min(width, height));
    const float force_scale = 0.125f * scale / static_cast<float>(rms_force);

    SDL_SetRenderDrawColor(sdl_renderer_handle, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(sdl_renderer_handle);

    SDL_Time current_time;
    SDL_GetCurrentTime(&current_time);
    last_draw_duration = current_time - last_draw_time;
    last_draw_time = current_time;

    angle += 1.0e-10 * static_cast<double>(angular_velocity) *
             static_cast<double>(last_draw_duration);
    if (angle >= PI) { angle -= (PI + PI); }
    if (angle < -PI) { angle += (PI + PI); }

    SDL_LockRWLockForReading(point_force_data_lock);
    renderer->import_points(points_x, points_y, points_z);
    if (render_forces) {
        renderer->import_forces(forces_x, forces_y, forces_z);
    }
    SDL_UnlockRWLock(point_force_data_lock);
    SDL_LockRWLockForReading(face_data_lock);
    renderer->import_faces(faces);
    SDL_UnlockRWLock(face_data_lock);

    if (render_neighbors) { renderer->compute_nearest_neighbor_colors(); }
    renderer->compute_screen_points(angle, origin_x, origin_y, scale);
    if (render_forces) { renderer->compute_screen_forces(angle, force_scale); }

    const float base_scale = SDL_GetWindowPixelDensity(sdl_window_handle);
    SDL_SetRenderScale(sdl_renderer_handle, base_scale, base_scale);

    if (render_neighbors) {
        SDL_SetRenderDrawColor(
            sdl_renderer_handle, 127, 127, 127, SDL_ALPHA_OPAQUE
        );
        renderer->render_nearest_neighbor_mesh(sdl_renderer_handle);
    }
    const int num_rendered_points =
        renderer->render_points(sdl_renderer_handle);
    if (render_forces) {
        SDL_SetRenderDrawColor(
            sdl_renderer_handle, 255, 255, 255, SDL_ALPHA_OPAQUE
        );
        renderer->render_forces(sdl_renderer_handle);
    }

    float debug_x = 0.0f;
    float debug_y = 0.0f;
    char debug_message_buffer[256];
    SDL_SetRenderScale(
        sdl_renderer_handle, 2.0f * base_scale, 2.0f * base_scale
    );
    SDL_SetRenderDrawColor(
        sdl_renderer_handle, 255, 255, 255, SDL_ALPHA_OPAQUE
    );

#define DEBUG_PRINT(format, ...)                                               \
    do {                                                                       \
        std::snprintf(                                                         \
            debug_message_buffer,                                              \
            sizeof(debug_message_buffer),                                      \
            format,                                                            \
            __VA_ARGS__                                                        \
        );                                                                     \
        SDL_RenderDebugText(                                                   \
            sdl_renderer_handle, debug_x, debug_y, debug_message_buffer        \
        );                                                                     \
        debug_y += 10.0f;                                                      \
    } while (0)
    DEBUG_PRINT("FPS:%5.0f", 1.0e9 / static_cast<double>(last_draw_duration));
    DEBUG_PRINT("Iteration count: %d", num_iterations.load());
    DEBUG_PRINT("Coulomb energy:  %.15e", energy.load());
    DEBUG_PRINT("RMS force:       %.15e", rms_force.load());
    DEBUG_PRINT("Step size:       %.15e", last_step_size.load());
    DEBUG_PRINT("RMS step length: %.15e", rms_step_length.load());
    DEBUG_PRINT(
        "Angular velocity: %+.1f rad/s",
        0.1 * static_cast<double>(angular_velocity)
    );
    DEBUG_PRINT(
        "Step time:%9.3f ms", 1.0e-6 * static_cast<double>(last_step_duration)
    );
    DEBUG_PRINT(
        "Hull time:%9.3f ms", 1.0e-6 * static_cast<double>(last_hull_duration)
    );
    DEBUG_PRINT(
        "Draw time:%9.3f ms", 1.0e-6 * static_cast<double>(last_draw_duration)
    );
    DEBUG_PRINT("Points drawn:%6d", num_rendered_points);
#undef DEBUG_PRINT

    SDL_RenderPresent(sdl_renderer_handle);
    return SDL_APP_CONTINUE;
}


void SDL_AppQuit(void *, SDL_AppResult) {
    using namespace GlobalVariables;
    quit = true;
    if (convex_hull_thread) { SDL_WaitThread(convex_hull_thread, nullptr); }
    if (optimizer_thread) { SDL_WaitThread(optimizer_thread, nullptr); }
    if (renderer) { delete renderer; }
    if (optimizer) { delete optimizer; }
    if (face_data_lock) { SDL_DestroyRWLock(face_data_lock); }
    if (point_force_data_lock) { SDL_DestroyRWLock(point_force_data_lock); }
    if (sdl_renderer_handle) { SDL_DestroyRenderer(sdl_renderer_handle); }
    if (sdl_window_handle) { SDL_DestroyWindow(sdl_window_handle); }
    if (faces) { delete[] faces; }
    if (forces_z) { delete[] forces_z; }
    if (forces_y) { delete[] forces_y; }
    if (forces_x) { delete[] forces_x; }
    if (points_z) { delete[] points_z; }
    if (points_y) { delete[] points_y; }
    if (points_x) { delete[] points_x; }
}
