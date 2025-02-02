#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_log.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_time.h>


static float rand_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}


static SDL_FColor random_color() {
    return {rand_float(), rand_float(), rand_float(), SDL_ALPHA_OPAQUE_FLOAT};
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
constexpr int NUM_CIRCLE_VECTORS = 48;
constexpr SDL_FPoint CIRCLE_VECTORS[NUM_CIRCLE_VECTORS] = {
    {+0.991444861f, +0.130526192f}, {+0.965925826f, +0.258819045f},
    {+0.923879533f, +0.382683432f}, {+0.866025404f, +0.500000000f},
    {+0.793353340f, +0.608761429f}, {+0.707106781f, +0.707106781f},
    {+0.608761429f, +0.793353340f}, {+0.500000000f, +0.866025404f},
    {+0.382683432f, +0.923879533f}, {+0.258819045f, +0.965925826f},
    {+0.130526192f, +0.991444861f}, {+0.000000000f, +1.000000000f},
    {-0.130526192f, +0.991444861f}, {-0.258819045f, +0.965925826f},
    {-0.382683432f, +0.923879533f}, {-0.500000000f, +0.866025404f},
    {-0.608761429f, +0.793353340f}, {-0.707106781f, +0.707106781f},
    {-0.793353340f, +0.608761429f}, {-0.866025404f, +0.500000000f},
    {-0.923879533f, +0.382683432f}, {-0.965925826f, +0.258819045f},
    {-0.991444861f, +0.130526192f}, {-1.000000000f, +0.000000000f},
    {-0.991444861f, -0.130526192f}, {-0.965925826f, -0.258819045f},
    {-0.923879533f, -0.382683432f}, {-0.866025404f, -0.500000000f},
    {-0.793353340f, -0.608761429f}, {-0.707106781f, -0.707106781f},
    {-0.608761429f, -0.793353340f}, {-0.500000000f, -0.866025404f},
    {-0.382683432f, -0.923879533f}, {-0.258819045f, -0.965925826f},
    {-0.130526192f, -0.991444861f}, {+0.000000000f, -1.000000000f},
    {+0.130526192f, -0.991444861f}, {+0.258819045f, -0.965925826f},
    {+0.382683432f, -0.923879533f}, {+0.500000000f, -0.866025404f},
    {+0.608761429f, -0.793353340f}, {+0.707106781f, -0.707106781f},
    {+0.793353340f, -0.608761429f}, {+0.866025404f, -0.500000000f},
    {+0.923879533f, -0.382683432f}, {+0.965925826f, -0.258819045f},
    {+0.991444861f, -0.130526192f}, {+1.000000000f, +0.000000000f},
};


static void construct_circle_vertices(
    SDL_Vertex *vertices, SDL_FPoint center, float radius, SDL_FColor color
) {
    SDL_Vertex c, v, w;
    c.position = center;
    c.color = color;
    v.color = color;
    w.position = {center.x + radius, center.y};
    w.color = color;
    for (int i = 0; i < NUM_CIRCLE_VECTORS; i++) {
        const SDL_FPoint circle_vector = CIRCLE_VECTORS[i];
        v.position = w.position;
        w.position = {
            std::fmaf(circle_vector.x, radius, center.x),
            std::fmaf(circle_vector.y, radius, center.y),
        };
        vertices[3 * i + 0] = c;
        vertices[3 * i + 1] = v;
        vertices[3 * i + 2] = w;
    }
}


using real_t = double;
constexpr real_t ZERO = static_cast<real_t>(0);
constexpr real_t ONE = static_cast<real_t>(1);


static void compute_coulomb_gradient(
    real_t *gradient, const real_t *points, int num_points
) {
    using std::sqrt;
    for (int i = 0; i < 3 * num_points; i++) { gradient[i] = ZERO; }
    for (int j = 1; j < num_points; j++) {
        const real_t xj = points[3 * j + 0];
        const real_t yj = points[3 * j + 1];
        const real_t zj = points[3 * j + 2];
        for (int i = 0; i < j; i++) {
            const real_t xi = points[3 * i + 0];
            const real_t yi = points[3 * i + 1];
            const real_t zi = points[3 * i + 2];
            const real_t dx = xi - xj;
            const real_t dy = yi - yj;
            const real_t dz = zi - zj;
            const real_t norm_squared = dx * dx + dy * dy + dz * dz;
            const real_t inv_norm_cubed =
                ONE / (norm_squared * sqrt(norm_squared));
            const real_t fx = dx * inv_norm_cubed;
            const real_t fy = dy * inv_norm_cubed;
            const real_t fz = dz * inv_norm_cubed;
            gradient[3 * i + 0] -= fx;
            gradient[3 * i + 1] -= fy;
            gradient[3 * i + 2] -= fz;
            gradient[3 * j + 0] += fx;
            gradient[3 * j + 1] += fy;
            gradient[3 * j + 2] += fz;
        }
    }
}


static void move_points(
    real_t *points, const real_t *gradient, real_t step_size, int num_points
) {
    using std::sqrt;
    for (int i = 0; i < num_points; i++) {
        const real_t x = points[3 * i + 0] - step_size * gradient[3 * i + 0];
        const real_t y = points[3 * i + 1] - step_size * gradient[3 * i + 1];
        const real_t z = points[3 * i + 2] - step_size * gradient[3 * i + 2];
        const real_t norm_squared = x * x + y * y + z * z;
        const real_t inv_norm = ONE / sqrt(norm_squared);
        points[3 * i + 0] = x * inv_norm;
        points[3 * i + 1] = y * inv_norm;
        points[3 * i + 2] = z * inv_norm;
    }
}


static void
quantize_points(double *points, const real_t *precise_points, int num_points) {
    for (int i = 0; i < num_points; i++) {
        points[3 * i + 0] = static_cast<double>(precise_points[3 * i + 0]);
        points[3 * i + 1] = static_cast<double>(precise_points[3 * i + 1]);
        points[3 * i + 2] = static_cast<double>(precise_points[3 * i + 2]);
    }
}


static void
quantize_forces(double *forces, const real_t *gradient, int num_points) {
    for (int i = 0; i < num_points; i++) {
        forces[3 * i + 0] = -static_cast<double>(gradient[3 * i + 0]);
        forces[3 * i + 1] = -static_cast<double>(gradient[3 * i + 1]);
        forces[3 * i + 2] = -static_cast<double>(gradient[3 * i + 2]);
    }
}


namespace GlobalVariables {

constexpr int INITIAL_WINDOW_WIDTH = 1920;
constexpr int INITIAL_WINDOW_HEIGHT = 1080;

static int num_points = 0;
static SDL_Time last_draw_time = 0;
static double angle = 0.0;
static double angular_velocity = 0.0;

static SDL_Window *window = nullptr;
static SDL_Renderer *renderer = nullptr;
static real_t *optimizer_points = nullptr;
static real_t *optimizer_gradient = nullptr;
static SDL_RWLock *renderer_lock = nullptr;
static double *renderer_points = nullptr;
static double *renderer_forces = nullptr;
static SDL_FColor *renderer_colors = nullptr;
static SDL_Vertex *renderer_vertices = nullptr;

} // namespace GlobalVariables


SDL_AppResult SDL_AppInit(void **, int, char **) {

    using namespace GlobalVariables;

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

    num_points = 2000;

    optimizer_points = static_cast<real_t *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(real_t))
    );
    if (!optimizer_points) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    optimizer_gradient = static_cast<real_t *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(real_t))
    );
    if (!optimizer_gradient) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    renderer_lock = SDL_CreateRWLock();
    if (!renderer_lock) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create lock.\n");
        return SDL_APP_FAILURE;
    }

    renderer_points = static_cast<double *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(double))
    );
    if (!renderer_points) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    renderer_forces = static_cast<double *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(double))
    );
    if (!renderer_forces) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    renderer_colors = static_cast<SDL_FColor *>(
        std::malloc(static_cast<std::size_t>(num_points) * sizeof(SDL_FColor))
    );
    if (!renderer_colors) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    renderer_vertices = static_cast<SDL_Vertex *>(std::malloc(
        static_cast<std::size_t>(num_points) * (3 * NUM_CIRCLE_VECTORS) *
        sizeof(SDL_Vertex)
    ));
    if (!renderer_vertices) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

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
        optimizer_points[3 * i + 0] = x;
        optimizer_points[3 * i + 1] = y;
        optimizer_points[3 * i + 2] = z;
        renderer_colors[i] = random_color();
    }

    compute_coulomb_gradient(optimizer_gradient, optimizer_points, num_points);
    quantize_points(renderer_points, optimizer_points, num_points);
    quantize_forces(renderer_forces, optimizer_gradient, num_points);

    SDL_GetCurrentTime(&last_draw_time);

    return SDL_APP_CONTINUE;
}


SDL_AppResult SDL_AppEvent(void *, SDL_Event *event) {
    using GlobalVariables::angular_velocity;
    switch (event->type) {
        case SDL_EVENT_QUIT: return SDL_APP_SUCCESS;
        case SDL_EVENT_KEY_DOWN: {
            if (event->key.key == SDLK_ESCAPE) { return SDL_APP_SUCCESS; }
            if (event->key.key == SDLK_LEFT) { angular_velocity -= 2.0e-10; }
            if (event->key.key == SDLK_RIGHT) { angular_velocity += 2.0e-10; }
            return SDL_APP_CONTINUE;
        }
        default: return SDL_APP_CONTINUE;
    }
}


SDL_AppResult SDL_AppIterate(void *) {

    using namespace GlobalVariables;

    move_points(optimizer_points, optimizer_gradient, 1.0e-6, num_points);
    compute_coulomb_gradient(optimizer_gradient, optimizer_points, num_points);

    SDL_LockRWLockForWriting(renderer_lock);
    quantize_points(renderer_points, optimizer_points, num_points);
    quantize_forces(renderer_forces, optimizer_gradient, num_points);
    SDL_UnlockRWLock(renderer_lock);

    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    const float origin_x = 0.5f * static_cast<float>(width);
    const float origin_y = 0.5f * static_cast<float>(height);
    const float scale = 0.375f * static_cast<float>(std::min(width, height));

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    SDL_Time time;
    SDL_GetCurrentTime(&time);
    const SDL_Time frame_duration = time - last_draw_time;
    angle += angular_velocity * static_cast<double>(frame_duration);
    if (angle >= PI) { angle -= (PI + PI); }
    if (angle < -PI) { angle += (PI + PI); }
    last_draw_time = time;
    const double sin_angle = std::sin(angle);
    const double cos_angle = std::cos(angle);

    int num_rendered_points = 0;
    SDL_Vertex *vertex_pointer = renderer_vertices;
    SDL_LockRWLockForReading(renderer_lock);
    for (int i = 0; i < num_points; i++) {
        // Transform world space to view space in double precision.
        const double x = static_cast<double>(renderer_points[3 * i + 0]);
        const double y = static_cast<double>(renderer_points[3 * i + 1]);
        const double z = static_cast<double>(renderer_points[3 * i + 2]);
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
            vertex_pointer += 3 * NUM_CIRCLE_VECTORS;
        }
    }
    SDL_UnlockRWLock(renderer_lock);

    SDL_RenderGeometry(
        renderer,
        nullptr,
        renderer_vertices,
        3 * NUM_CIRCLE_VECTORS * num_rendered_points,
        nullptr,
        0
    );

    char debug_message_buffer[256];
    SDL_SetRenderScale(renderer, 2.0f, 2.0f);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "FPS:%6.0f",
        1.0e9 / static_cast<double>(frame_duration)
    );
    SDL_RenderDebugText(renderer, 0.0f, 0.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Frame time:%9.3f ms",
        1.0e-6 * static_cast<double>(frame_duration)
    );
    SDL_RenderDebugText(renderer, 0.0f, 10.0f, debug_message_buffer);
    std::snprintf(
        debug_message_buffer,
        sizeof(debug_message_buffer),
        "Points drawn:%7d",
        num_rendered_points
    );
    SDL_RenderDebugText(renderer, 0.0f, 20.0f, debug_message_buffer);
    SDL_SetRenderScale(renderer, 1.0f, 1.0f);

    SDL_RenderPresent(renderer);

    return SDL_APP_CONTINUE;
}


void SDL_AppQuit(void *, SDL_AppResult) {
    using namespace GlobalVariables;
    if (renderer_vertices) { std::free(renderer_vertices); }
    if (renderer_colors) { std::free(renderer_colors); }
    if (renderer_forces) { std::free(renderer_forces); }
    if (renderer_points) { std::free(renderer_points); }
    if (renderer_lock) { SDL_DestroyRWLock(renderer_lock); }
    if (optimizer_gradient) { std::free(optimizer_gradient); }
    if (optimizer_points) { std::free(optimizer_points); }
    if (renderer) { SDL_DestroyRenderer(renderer); }
    if (window) { SDL_DestroyWindow(window); }
}
