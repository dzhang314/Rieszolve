#include <cmath>
#include <cstdio>
#include <cstdlib>

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>


constexpr SDL_FColor YELLOW = {1.0f, 1.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT};


static float rand_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}


static SDL_FColor random_color() {
    return {rand_float(), rand_float(), rand_float(), SDL_ALPHA_OPAQUE_FLOAT};
}


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


static void
construct_circle_points(SDL_FPoint *points, SDL_FPoint center, float radius) {
    for (int i = 0; i < NUM_CIRCLE_VECTORS; i++) {
        const SDL_FPoint circle_vector = CIRCLE_VECTORS[i];
        points[i] = {
            std::fmaf(circle_vector.x, radius, center.x),
            std::fmaf(circle_vector.y, radius, center.y),
        };
    }
}


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
static SDL_Time init_time = 0;

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
        printf("Failed to create window: %s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    num_points = 2000;

    points = static_cast<real_t *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(real_t))
    );
    if (!points) {
        printf("Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    forces = static_cast<real_t *>(
        std::malloc(3 * static_cast<std::size_t>(num_points) * sizeof(real_t))
    );
    if (!forces) {
        printf("Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    colors = static_cast<SDL_FColor *>(
        std::malloc(static_cast<std::size_t>(num_points) * sizeof(SDL_FColor))
    );
    if (!colors) {
        printf("Failed to allocate memory.\n");
        return SDL_APP_FAILURE;
    }

    vertices = static_cast<SDL_Vertex *>(std::malloc(
        static_cast<std::size_t>(num_points) * (3 * NUM_CIRCLE_VECTORS) *
        sizeof(SDL_Vertex)
    ));
    if (!vertices) {
        printf("Failed to allocate memory.\n");
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

    SDL_GetCurrentTime(&init_time);

    return SDL_APP_CONTINUE;
}


SDL_AppResult SDL_AppEvent(void *, SDL_Event *event) {
    switch (event->type) {
        case SDL_EVENT_QUIT: return SDL_APP_SUCCESS;
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
    const float theta = 0.3e-9f * static_cast<float>(time - init_time);

    int k = 0;
    int num_drawn_points = 0;
    SDL_Vertex *vertex_pointer = vertices;
    for (int i = 0; i < num_points; i++) {
        const float x = static_cast<float>(points[k++]);
        const float y = static_cast<float>(points[k++]);
        const float z = static_cast<float>(points[k++]);
        const float rx = x * std::cosf(theta) + z * std::sinf(theta);
        const float ry = y;
        const float rz = z * std::cosf(theta) - x * std::sinf(theta);
        if (rz >= 0.0f) {
            ++num_drawn_points;
            const float screen_x = origin_x + scale * static_cast<float>(rx);
            const float screen_y = origin_y - scale * static_cast<float>(ry);
            construct_circle_vertices(
                vertex_pointer,
                {screen_x, screen_y},
                3.0f * rz + 1.0f,
                colors[i]
            );
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
