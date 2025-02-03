#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>


static float rand_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}


static SDL_FColor random_color() {
    const float r = rand_float();
    const float g = rand_float();
    const float b = rand_float();
    const float scale = 1.0f / std::max(std::max(r, g), b);
    return {scale * r, scale * g, scale * b, SDL_ALPHA_OPAQUE_FLOAT};
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
constexpr int NUM_CIRCLE_VERTICES = 36;


static inline void construct_circle_vertices(
    SDL_Vertex *vertices, SDL_FPoint center, float radius, SDL_FColor color
) {
    const float major = 0.866025404f * radius;
    const float minor = 0.5f * radius;

    vertices[0].position = center;
    vertices[0].color = color;
    vertices[1].position.x = center.x + radius;
    vertices[1].position.y = center.y;
    vertices[1].color = color;
    vertices[2].position.x = center.x + major;
    vertices[2].position.y = center.y + minor;
    vertices[2].color = color;

    vertices[3].position = center;
    vertices[3].color = color;
    vertices[4].position.x = center.x + major;
    vertices[4].position.y = center.y + minor;
    vertices[4].color = color;
    vertices[5].position.x = center.x + minor;
    vertices[5].position.y = center.y + major;
    vertices[5].color = color;

    vertices[6].position = center;
    vertices[6].color = color;
    vertices[7].position.x = center.x + minor;
    vertices[7].position.y = center.y + major;
    vertices[7].color = color;
    vertices[8].position.x = center.x;
    vertices[8].position.y = center.y + radius;
    vertices[8].color = color;

    vertices[9].position = center;
    vertices[9].color = color;
    vertices[10].position.x = center.x;
    vertices[10].position.y = center.y + radius;
    vertices[10].color = color;
    vertices[11].position.x = center.x - minor;
    vertices[11].position.y = center.y + major;
    vertices[11].color = color;

    vertices[12].position = center;
    vertices[12].color = color;
    vertices[13].position.x = center.x - minor;
    vertices[13].position.y = center.y + major;
    vertices[13].color = color;
    vertices[14].position.x = center.x - major;
    vertices[14].position.y = center.y + minor;
    vertices[14].color = color;

    vertices[15].position = center;
    vertices[15].color = color;
    vertices[16].position.x = center.x - major;
    vertices[16].position.y = center.y + minor;
    vertices[16].color = color;
    vertices[17].position.x = center.x - radius;
    vertices[17].position.y = center.y;
    vertices[17].color = color;

    vertices[18].position = center;
    vertices[18].color = color;
    vertices[19].position.x = center.x - radius;
    vertices[19].position.y = center.y;
    vertices[19].color = color;
    vertices[20].position.x = center.x - major;
    vertices[20].position.y = center.y - minor;
    vertices[20].color = color;

    vertices[21].position = center;
    vertices[21].color = color;
    vertices[22].position.x = center.x - major;
    vertices[22].position.y = center.y - minor;
    vertices[22].color = color;
    vertices[23].position.x = center.x - minor;
    vertices[23].position.y = center.y - major;
    vertices[23].color = color;

    vertices[24].position = center;
    vertices[24].color = color;
    vertices[25].position.x = center.x - minor;
    vertices[25].position.y = center.y - major;
    vertices[25].color = color;
    vertices[26].position.x = center.x;
    vertices[26].position.y = center.y - radius;
    vertices[26].color = color;

    vertices[27].position = center;
    vertices[27].color = color;
    vertices[28].position.x = center.x;
    vertices[28].position.y = center.y - radius;
    vertices[28].color = color;
    vertices[29].position.x = center.x + minor;
    vertices[29].position.y = center.y - major;
    vertices[29].color = color;

    vertices[30].position = center;
    vertices[30].color = color;
    vertices[31].position.x = center.x + minor;
    vertices[31].position.y = center.y - major;
    vertices[31].color = color;
    vertices[32].position.x = center.x + major;
    vertices[32].position.y = center.y - minor;
    vertices[32].color = color;

    vertices[33].position = center;
    vertices[33].color = color;
    vertices[34].position.x = center.x + major;
    vertices[34].position.y = center.y - minor;
    vertices[34].color = color;
    vertices[35].position.x = center.x + radius;
    vertices[35].position.y = center.y;
    vertices[35].color = color;
}


using real_t = double;
constexpr real_t ZERO = static_cast<real_t>(0);
constexpr real_t ONE = static_cast<real_t>(1);


static void compute_coulomb_gradient(
    real_t *gradient_x,
    real_t *gradient_y,
    real_t *gradient_z,
    const real_t *points_x,
    const real_t *points_y,
    const real_t *points_z,
    int num_points
) {
    using std::sqrt;
#pragma omp parallel for schedule(static) default(none)                        \
    shared(gradient_x, gradient_y, gradient_z)                                 \
    shared(points_x, points_y, points_z, num_points)
    for (int i = 0; i < num_points; i++) {
        const real_t xi = points_x[i];
        const real_t yi = points_y[i];
        const real_t zi = points_z[i];
        real_t gx = ZERO;
        real_t gy = ZERO;
        real_t gz = ZERO;
#pragma omp simd reduction(+ : gx, gy, gz) simdlen(8)                          \
    aligned(gradient_x, gradient_y, gradient_z : 64)                           \
    aligned(points_x, points_y, points_z : 64)
        for (int j = 0; j < num_points; j++) {
            if (i != j) {
                const real_t dx = points_x[j] - xi;
                const real_t dy = points_y[j] - yi;
                const real_t dz = points_z[j] - zi;
                const real_t norm_squared = dx * dx + dy * dy + dz * dz;
                const real_t inv_norm_cubed =
                    ONE / (norm_squared * sqrt(norm_squared));
                gx += dx * inv_norm_cubed;
                gy += dy * inv_norm_cubed;
                gz += dz * inv_norm_cubed;
            }
        }
        gradient_x[i] = gx;
        gradient_y[i] = gy;
        gradient_z[i] = gz;
    }
}


static void move_points(
    real_t *points_x,
    real_t *points_y,
    real_t *points_z,
    const real_t *step_direction_x,
    const real_t *step_direction_y,
    const real_t *step_direction_z,
    real_t step_size,
    int num_points
) {
    using std::sqrt;
#pragma omp simd simdlen(8) aligned(points_x, points_y, points_z : 64)         \
    aligned(step_direction_x, step_direction_y, step_direction_z : 64)
    for (int i = 0; i < num_points; i++) {
        const real_t x = points_x[i] - step_size * step_direction_x[i];
        const real_t y = points_y[i] - step_size * step_direction_y[i];
        const real_t z = points_z[i] - step_size * step_direction_z[i];
        const real_t norm_squared = x * x + y * y + z * z;
        const real_t inv_norm = ONE / sqrt(norm_squared);
        points_x[i] = x * inv_norm;
        points_y[i] = y * inv_norm;
        points_z[i] = z * inv_norm;
    }
}


static void quantize_points(
    double *points,
    const real_t *points_x,
    const real_t *points_y,
    const real_t *points_z,
    int num_points
) {
#pragma omp simd aligned(points_x, points_y, points_z : 64) simdlen(8)
    for (int i = 0; i < num_points; i++) {
        points[3 * i + 0] = static_cast<double>(points_x[i]);
        points[3 * i + 1] = static_cast<double>(points_y[i]);
        points[3 * i + 2] = static_cast<double>(points_z[i]);
    }
}


static void quantize_forces(
    double *forces,
    const real_t *gradient_x,
    const real_t *gradient_y,
    const real_t *gradient_z,
    int num_points
) {
#pragma omp simd aligned(gradient_x, gradient_y, gradient_z : 64) simdlen(8)
    for (int i = 0; i < num_points; i++) {
        forces[3 * i + 0] = -static_cast<double>(gradient_x[i]);
        forces[3 * i + 1] = -static_cast<double>(gradient_y[i]);
        forces[3 * i + 2] = -static_cast<double>(gradient_z[i]);
    }
}


namespace GlobalVariables {

constexpr int INITIAL_WINDOW_WIDTH = 1920;
constexpr int INITIAL_WINDOW_HEIGHT = 1080;

static int num_points = 0;
static int num_steps = 0;
static SDL_Time last_draw_time = 0;
static SDL_Time last_step_time = 0;
static SDL_Time last_draw_duration = 0;
static SDL_Time last_step_duration = 0;
static double angle = 0.0;
static int angular_velocity = 2;
static bool quit = false;

static SDL_Window *window = nullptr;
static SDL_Renderer *renderer = nullptr;
static real_t *optimizer_points_x = nullptr;
static real_t *optimizer_points_y = nullptr;
static real_t *optimizer_points_z = nullptr;
static real_t *optimizer_gradient_x = nullptr;
static real_t *optimizer_gradient_y = nullptr;
static real_t *optimizer_gradient_z = nullptr;
static SDL_RWLock *renderer_lock = nullptr;
static double *renderer_points = nullptr;
static double *renderer_forces = nullptr;
static SDL_FColor *renderer_colors = nullptr;
static SDL_Vertex *renderer_vertices = nullptr;
static SDL_Thread *optimizer_thread = nullptr;

} // namespace GlobalVariables


static int SDLCALL run_optimizer(void *) {
    using namespace GlobalVariables;
    while (!quit) {

        move_points(
            optimizer_points_x,
            optimizer_points_y,
            optimizer_points_z,
            optimizer_gradient_x,
            optimizer_gradient_y,
            optimizer_gradient_z,
            1.0e-7,
            num_points
        );
        compute_coulomb_gradient(
            optimizer_gradient_x,
            optimizer_gradient_y,
            optimizer_gradient_z,
            optimizer_points_x,
            optimizer_points_y,
            optimizer_points_z,
            num_points
        );

        SDL_LockRWLockForWriting(renderer_lock);
        quantize_points(
            renderer_points,
            optimizer_points_x,
            optimizer_points_y,
            optimizer_points_z,
            num_points
        );
        quantize_forces(
            renderer_forces,
            optimizer_gradient_x,
            optimizer_gradient_y,
            optimizer_gradient_z,
            num_points
        );
        SDL_UnlockRWLock(renderer_lock);

        num_steps++;
        SDL_Time current_time;
        SDL_GetCurrentTime(&current_time);
        last_step_duration = current_time - last_step_time;
        last_step_time = current_time;
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


#define FREE_MEMORY(POINTER_NAME)                                              \
    do {                                                                       \
        if (POINTER_NAME) { std::free(POINTER_NAME); }                         \
        POINTER_NAME = nullptr;                                                \
    } while (false)


#define FREE_ALIGNED_MEMORY(POINTER_NAME)                                      \
    do {                                                                       \
        if (POINTER_NAME) { _aligned_free(POINTER_NAME); }                     \
        POINTER_NAME = nullptr;                                                \
    } while (false)


SDL_AppResult SDL_AppInit(void **, int, char **) {

    using namespace GlobalVariables;

    num_points = 2000;
    SDL_GetCurrentTime(&last_draw_time);
    SDL_GetCurrentTime(&last_step_time);

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

    ALLOCATE_ALIGNED_MEMORY(optimizer_points_x, real_t, num_points);
    ALLOCATE_ALIGNED_MEMORY(optimizer_points_y, real_t, num_points);
    ALLOCATE_ALIGNED_MEMORY(optimizer_points_z, real_t, num_points);
    ALLOCATE_ALIGNED_MEMORY(optimizer_gradient_x, real_t, num_points);
    ALLOCATE_ALIGNED_MEMORY(optimizer_gradient_y, real_t, num_points);
    ALLOCATE_ALIGNED_MEMORY(optimizer_gradient_z, real_t, num_points);

    renderer_lock = SDL_CreateRWLock();
    if (!renderer_lock) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create lock.\n");
        return SDL_APP_FAILURE;
    }

    ALLOCATE_MEMORY(renderer_points, double, 3 * num_points);
    ALLOCATE_MEMORY(renderer_forces, double, 3 * num_points);
    ALLOCATE_MEMORY(renderer_colors, SDL_FColor, num_points);
    ALLOCATE_MEMORY(
        renderer_vertices, SDL_Vertex, NUM_CIRCLE_VERTICES * num_points
    );

    for (int i = 0; i < num_points; i++) {
        real_t x, y, z;
        while (true) {
            x = static_cast<real_t>(rand_float());
            x = x + x - ONE;
            y = static_cast<real_t>(rand_float());
            y = y + y - ONE;
            z = static_cast<real_t>(rand_float());
            z = z + z - ONE;
            const real_t norm_squared = x * x + y * y + z * z;
            if (norm_squared <= ONE) {
                using std::sqrt;
                const real_t norm = sqrt(norm_squared);
                const real_t inv_norm = ONE / norm;
                x *= inv_norm;
                y *= inv_norm;
                z *= inv_norm;
                break;
            }
        }
        optimizer_points_x[i] = x;
        optimizer_points_y[i] = y;
        optimizer_points_z[i] = z;
        renderer_colors[i] = random_color();
    }

    compute_coulomb_gradient(
        optimizer_gradient_x,
        optimizer_gradient_y,
        optimizer_gradient_z,
        optimizer_points_x,
        optimizer_points_y,
        optimizer_points_z,
        num_points
    );
    quantize_points(
        renderer_points,
        optimizer_points_x,
        optimizer_points_y,
        optimizer_points_z,
        num_points
    );
    quantize_forces(
        renderer_forces,
        optimizer_gradient_x,
        optimizer_gradient_y,
        optimizer_gradient_z,
        num_points
    );

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
    last_draw_duration = current_time - last_draw_time;
    last_step_duration = current_time - last_step_time;

    return SDL_APP_CONTINUE;
}


SDL_AppResult SDL_AppEvent(void *, SDL_Event *event) {
    using GlobalVariables::angular_velocity;
    switch (event->type) {
        case SDL_EVENT_QUIT: return SDL_APP_SUCCESS;
        case SDL_EVENT_KEY_DOWN: {
            switch (event->key.key) {
                case SDLK_ESCAPE: return SDL_APP_SUCCESS;
                case SDLK_Q: return SDL_APP_SUCCESS;
                case SDLK_LEFT: angular_velocity -= 2; break;
                case SDLK_RIGHT: angular_velocity += 2; break;
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
    SDL_GetWindowSize(window, &width, &height);
    const float origin_x = 0.5f * static_cast<float>(width);
    const float origin_y = 0.5f * static_cast<float>(height);
    const float scale = 0.375f * static_cast<float>(std::min(width, height));

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

    int num_rendered_points = 0;
    SDL_Vertex *vertex_pointer = renderer_vertices;
    SDL_LockRWLockForReading(renderer_lock);
    for (int i = 0; i < num_points; i++) {
        // Transform world space to view space in double precision.
        const double x = renderer_points[3 * i + 0];
        const double y = renderer_points[3 * i + 1];
        const double z = renderer_points[3 * i + 2];
        const float vx = static_cast<float>(x * cos_angle + z * sin_angle);
        const float vy = static_cast<float>(y);
        const float vz = static_cast<float>(z * cos_angle - x * sin_angle);
        if (vz >= 0.0f) {
            // Transform view space to screen space in single precision.
            ++num_rendered_points;
            const float sx = origin_x + scale * vx;
            const float sy = origin_y - scale * vy;
            // Simulate perspective by making closer points larger.
            // We should use a proper perspective transformation
            // in the future, but this is good enough for now.
            const float r = 3.0f * vz + 1.0f;
            construct_circle_vertices(
                vertex_pointer, {sx, sy}, r, renderer_colors[i]
            );
            vertex_pointer += NUM_CIRCLE_VERTICES;
        }
    }
    SDL_UnlockRWLock(renderer_lock);

    SDL_RenderGeometry(
        renderer,
        nullptr,
        renderer_vertices,
        NUM_CIRCLE_VERTICES * num_rendered_points,
        nullptr,
        0
    );

    char debug_message_buffer[256];
    SDL_SetRenderScale(renderer, 2.0f, 2.0f);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
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
        "Step count:%8d",
        num_steps
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
    SDL_SetRenderScale(renderer, 1.0f, 1.0f);

    SDL_RenderPresent(renderer);

    return SDL_APP_CONTINUE;
}


void SDL_AppQuit(void *, SDL_AppResult) {
    using namespace GlobalVariables;
    quit = true;
    if (optimizer_thread) { SDL_WaitThread(optimizer_thread, nullptr); }
    FREE_MEMORY(renderer_vertices);
    FREE_MEMORY(renderer_colors);
    FREE_MEMORY(renderer_forces);
    FREE_MEMORY(renderer_points);
    if (renderer_lock) { SDL_DestroyRWLock(renderer_lock); }
    FREE_ALIGNED_MEMORY(optimizer_gradient_z);
    FREE_ALIGNED_MEMORY(optimizer_gradient_y);
    FREE_ALIGNED_MEMORY(optimizer_gradient_x);
    FREE_ALIGNED_MEMORY(optimizer_points_z);
    FREE_ALIGNED_MEMORY(optimizer_points_y);
    FREE_ALIGNED_MEMORY(optimizer_points_x);
    if (renderer) { SDL_DestroyRenderer(renderer); }
    if (window) { SDL_DestroyWindow(window); }
}
