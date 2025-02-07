#pragma once


#include <SDL3/SDL_pixels.h> // for SDL_FColor
#include <SDL3/SDL_render.h> // for SDL_Vertex


constexpr int NUM_CIRCLE_VERTICES = 36;


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
