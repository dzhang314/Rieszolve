#pragma once


constexpr int INVALID_INDEX = -1;


struct HalfEdge {

    int vertex_index;
    int twin_index;

}; // struct HalfEdge


struct Triangle {

    HalfEdge ab;
    HalfEdge bc;
    HalfEdge ca;

    constexpr Triangle() noexcept
        : ab{INVALID_INDEX, INVALID_INDEX}
        , bc{INVALID_INDEX, INVALID_INDEX}
        , ca{INVALID_INDEX, INVALID_INDEX} {}

    constexpr Triangle(int a_index, int b_index, int c_index) noexcept
        : ab{a_index, INVALID_INDEX}
        , bc{b_index, INVALID_INDEX}
        , ca{c_index, INVALID_INDEX} {}

}; // struct Triangle


class TriangleMesh {

    Triangle *faces;
    int num_faces;

public:

    explicit TriangleMesh(
        const double *__restrict__ vertices_x,
        const double *__restrict__ vertices_y,
        const double *__restrict__ vertices_z,
        int num_vertices
    );
    TriangleMesh(const TriangleMesh &) = delete;
    TriangleMesh &operator=(const TriangleMesh &) = delete;
    ~TriangleMesh() noexcept;
    bool is_allocated() const noexcept;

    bool flip_edges(
        const double *__restrict__ vertices_x,
        const double *__restrict__ vertices_y,
        const double *__restrict__ vertices_z
    ) noexcept;

    Triangle get_face(int index) const noexcept;

}; // class TriangleMesh
