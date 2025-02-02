#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define SDL_MAIN_USE_CALLBACKS 1
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
    int k = 0;
    for (int i = 0; i < NUM_CIRCLE_VECTORS; i++) {
        const SDL_FPoint circle_vector = CIRCLE_VECTORS[i];
        v.position = w.position;
        w.position = {
            std::fmaf(circle_vector.x, radius, center.x),
            std::fmaf(circle_vector.y, radius, center.y),
        };
        vertices[k++] = c;
        vertices[k++] = v;
        vertices[k++] = w;
    }
}


constexpr int INITIAL_WINDOW_WIDTH = 1920;
constexpr int INITIAL_WINDOW_HEIGHT = 1080;
static SDL_Window *window = nullptr;
static SDL_Renderer *renderer = nullptr;
static SDL_Time last_draw_time = 0;
static double angle = 0.0;
static double angular_velocity = 0.0;

using real_t = double;
constexpr real_t ZERO = static_cast<real_t>(0);
constexpr real_t ONE = static_cast<real_t>(1);

static int num_points = 0;
static real_t *points = nullptr;
static real_t *forces = nullptr;
static SDL_FColor *colors = nullptr;
static SDL_Vertex *vertices = nullptr;


SDL_AppResult SDL_AppInit(void **, int, char **) {

    window = SDL_CreateWindow(
        "Rieszolve",
        INITIAL_WINDOW_WIDTH,
        INITIAL_WINDOW_HEIGHT,
        SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_RESIZABLE
    );
    if (!window) {
        std::printf("Failed to create window: %s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) {
        std::printf("Failed to create renderer: %s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    num_points = 2000;

    points = static_cast<real_t *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(real_t))
    );
    if (!points) {
        std::printf("Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    forces = static_cast<real_t *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(real_t))
    );
    if (!forces) {
        std::printf("Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    colors = static_cast<SDL_FColor *>(
        std::malloc(static_cast<std::size_t>(num_points) * sizeof(SDL_FColor))
    );
    if (!colors) {
        std::printf("Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    vertices = static_cast<SDL_Vertex *>(std::malloc(
        static_cast<std::size_t>(num_points) * (3 * NUM_CIRCLE_VECTORS) *
        sizeof(SDL_Vertex)
    ));
    if (!vertices) {
        std::printf("Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    int k = 0;
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
        points[k++] = x;
        points[k++] = y;
        points[k++] = z;
        colors[i] = random_color();
    }

    SDL_GetCurrentTime(&last_draw_time);

    return SDL_APP_CONTINUE;
}


SDL_AppResult SDL_AppEvent(void *, SDL_Event *event) {
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

static void compute_forces() {
    using std::sqrt;
    int k = 0;
    for (int i = 0; i < num_points; i++) {
        forces[k++] = ZERO;
        forces[k++] = ZERO;
        forces[k++] = ZERO;
    }
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
            forces[3 * i + 0] += dx * inv_norm_cubed;
            forces[3 * i + 1] += dy * inv_norm_cubed;
            forces[3 * i + 2] += dz * inv_norm_cubed;
            forces[3 * j + 0] -= dx * inv_norm_cubed;
            forces[3 * j + 1] -= dy * inv_norm_cubed;
            forces[3 * j + 2] -= dz * inv_norm_cubed;
        }
    }
}


static void move_points(real_t step_size) {
    using std::sqrt;
    for (int i = 0; i < num_points; i++) {
        real_t x = points[3 * i + 0] + step_size * forces[3 * i + 0];
        real_t y = points[3 * i + 1] + step_size * forces[3 * i + 1];
        real_t z = points[3 * i + 2] + step_size * forces[3 * i + 2];
        real_t norm_squared = x * x + y * y + z * z;
        real_t inv_norm = ONE / sqrt(norm_squared);
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
        points[3 * i + 0] = x;
        points[3 * i + 1] = y;
        points[3 * i + 2] = z;
    }
}


SDL_AppResult SDL_AppIterate(void *) {

    compute_forces();
    move_points(0.00001);

    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    const float origin_x = 0.5f * static_cast<float>(width);
    const float origin_y = 0.5f * static_cast<float>(height);
    const float scale = 0.375f * static_cast<float>(std::min(width, height));

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    SDL_Time time;
    SDL_GetCurrentTime(&time);
    angle += angular_velocity * static_cast<double>(time - last_draw_time);
    if (angle >= PI) { angle -= (PI + PI); }
    if (angle < -PI) { angle += (PI + PI); }
    last_draw_time = time;
    const double sin_angle = std::sin(angle);
    const double cos_angle = std::cos(angle);

    int k = 0;
    int num_drawn_points = 0;
    SDL_Vertex *vertex_pointer = vertices;
    for (int i = 0; i < num_points; i++) {
        // Transform world space to view space in double precision.
        const double x = static_cast<double>(points[k++]);
        const double y = static_cast<double>(points[k++]);
        const double z = static_cast<double>(points[k++]);
        const float vx = static_cast<float>(x * cos_angle + z * sin_angle);
        const float vy = static_cast<float>(y);
        const float vz = static_cast<float>(z * cos_angle - x * sin_angle);
        if (vz >= 0.0f) {
            // Transform view space to screen space in single precision.
            ++num_drawn_points;
            const float sx = origin_x + scale * vx;
            const float sy = origin_y - scale * vy;
            // Simulate perspective by making closer points larger.
            // We should use a proper perspective transformation
            // in the future, but this is good enough for now.
            const float r = 3.0f * vz + 1.0f;
            construct_circle_vertices(vertex_pointer, {sx, sy}, r, colors[i]);
            vertex_pointer += 3 * NUM_CIRCLE_VECTORS;
        }
    }

    SDL_RenderGeometry(
        renderer,
        nullptr,
        vertices,
        3 * NUM_CIRCLE_VECTORS * num_drawn_points,
        nullptr,
        0
    );
    SDL_RenderPresent(renderer);

    return SDL_APP_CONTINUE;
}


void SDL_AppQuit(void *, SDL_AppResult) {
    std::free(vertices);
    std::free(colors);
    std::free(forces);
    std::free(points);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
}
