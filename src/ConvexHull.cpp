#include "ConvexHull.hpp"

#include <array>
#include <cassert>
#include <set>
#include <utility>
#include <vector>


using REAL_T = double;
using INDEX_T = unsigned int;
constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(-1);


struct Vector3D {

    REAL_T x;
    REAL_T y;
    REAL_T z;

    constexpr Vector3D operator-(const Vector3D &rhs) const noexcept {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }

    constexpr REAL_T operator*(const Vector3D &rhs) const noexcept {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    constexpr Vector3D operator^(const Vector3D &rhs) const noexcept {
        return {
            y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z, x * rhs.y - y * rhs.x
        };
    }

}; // struct Vector3D


struct Plane3D {

    Vector3D normal;
    REAL_T distance;

    explicit constexpr Plane3D(
        const Vector3D &a, const Vector3D &b, const Vector3D &c
    ) noexcept
        : normal((b - a) ^ (c - a))
        , distance(normal * a) {}

    constexpr REAL_T operator()(const Vector3D &point) const noexcept {
        return normal * point - distance;
    }

}; // struct Plane3D


struct HalfEdge {

    INDEX_T origin_index;
    INDEX_T twin_index;
    INDEX_T face_index;
    INDEX_T next_index;

    void clear() noexcept { origin_index = INVALID_INDEX; }

    constexpr bool cleared() const noexcept {
        return (origin_index == INVALID_INDEX);
    }

}; // struct HalfEdge


struct Face {

    INDEX_T half_edge_index;

    void clear() noexcept { half_edge_index = INVALID_INDEX; }

    constexpr bool cleared() const noexcept {
        return (half_edge_index == INVALID_INDEX);
    }

}; // struct Face


struct Polyhedron {

    std::vector<HalfEdge> half_edges;
    std::vector<Face> faces;
    std::vector<INDEX_T> vacant_half_edge_indices;
    std::vector<INDEX_T> vacant_face_indices;

    INDEX_T add_empty_half_edge() {
        if (vacant_half_edge_indices.empty()) {
            half_edges.push_back(
                {INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX}
            );
            return static_cast<INDEX_T>(half_edges.size() - 1);
        } else {
            const INDEX_T half_edge_index = vacant_half_edge_indices.back();
            vacant_half_edge_indices.pop_back();
            assert(half_edges.at(half_edge_index).cleared());
            return half_edge_index;
        }
    }

    INDEX_T add_empty_face() {
        if (vacant_face_indices.empty()) {
            faces.emplace_back();
            return static_cast<INDEX_T>(faces.size() - 1);
        } else {
            const INDEX_T face_index = vacant_face_indices.back();
            vacant_face_indices.pop_back();
            assert(faces.at(face_index).cleared());
            return face_index;
        }
    }

    void evict_half_edge(INDEX_T half_edge_index) {
        half_edges.at(half_edge_index).clear();
        vacant_half_edge_indices.push_back(half_edge_index);
    }

    void evict_face(INDEX_T face_index) {
        faces.at(face_index).clear();
        vacant_face_indices.push_back(face_index);
    }

    std::array<INDEX_T, 2> get_vertex_indices(const HalfEdge &half_edge) const {
        return {
            half_edge.origin_index,
            half_edges.at(half_edge.twin_index).origin_index
        };
    }

    std::array<INDEX_T, 3> get_vertex_indices(const Face &face) const {
        std::array<INDEX_T, 3> result;
        HalfEdge half_edge = half_edges.at(face.half_edge_index);
        result[0] = half_edge.origin_index;
        half_edge = half_edges.at(half_edge.next_index);
        result[1] = half_edge.origin_index;
        half_edge = half_edges.at(half_edge.next_index);
        result[2] = half_edge.origin_index;
        return result;
    }

    std::array<INDEX_T, 3> get_half_edge_indices(const Face &face) const {
        std::array<INDEX_T, 3> result;
        result[0] = face.half_edge_index;
        result[1] = half_edges.at(result[0]).next_index;
        result[2] = half_edges.at(result[1]).next_index;
        return result;
    }

    Plane3D get_normal_plane(
        const Face &face, const std::vector<Vector3D> &points
    ) const {
        const auto vertex_indices = get_vertex_indices(face);
        const Vector3D a = points.at(vertex_indices[0]);
        const Vector3D b = points.at(vertex_indices[1]);
        const Vector3D c = points.at(vertex_indices[2]);
        return Plane3D{a, b, c};
    }

    // A list of half-edges is cycle-sorted if the target vertex
    // of each half-edge coincides with the origin vertex of the
    // next, with the last half-edge wrapping around to the first.
    bool cycle_sort_edges(std::vector<INDEX_T> &half_edge_indices) const {
        // The empty list is trivially cycle-sorted.
        if (half_edge_indices.empty()) { return true; }
        using vec_size_t = std::vector<INDEX_T>::size_type;
        const vec_size_t n = half_edge_indices.size();
        // This O(n^2) algorithm is suboptimal, but it is fast enough
        // for this application and avoids dynamic data structures.
        for (vec_size_t i = 0; i < n - 1; ++i) {
            const INDEX_T twin_index =
                half_edges.at(half_edge_indices[i]).twin_index;
            const INDEX_T target_index = half_edges.at(twin_index).origin_index;
            // Find a half-edge whose origin vertex coincides
            // with the target vertex of the current half-edge.
            bool found = false;
            for (vec_size_t j = i + 1; j < n; ++j) {
                const INDEX_T next_origin_index =
                    half_edges.at(half_edge_indices[j]).origin_index;
                if (next_origin_index == target_index) {
                    std::swap(half_edge_indices[i + 1], half_edge_indices[j]);
                    found = true;
                    break;
                }
            }
            if (!found) { return false; }
        }
        // Check that the last half-edge correctly wraps around to the first.
        const INDEX_T last_target_index =
            half_edges.at(half_edges.at(half_edge_indices[n - 1]).twin_index)
                .origin_index;
        const INDEX_T first_origin_index =
            half_edges.at(half_edge_indices[0]).origin_index;
        return (first_origin_index == last_target_index);
    }

    void find_visible_faces(
        std::vector<INDEX_T> &visible_face_indices,
        std::set<INDEX_T> &horizon_edge_indices,
        const std::vector<Vector3D> &points,
        const Vector3D &point
    ) const {
        visible_face_indices.clear();
        const INDEX_T num_faces = static_cast<INDEX_T>(faces.size());
        for (INDEX_T face_index = 0; face_index < num_faces; ++face_index) {
            const Face &face = faces[face_index];
            const Plane3D normal_plane = get_normal_plane(face, points);
            if (normal_plane(point) >= 0.0) {
                visible_face_indices.push_back(face_index);
                const auto half_edge_indices = get_half_edge_indices(face);
                for (const INDEX_T half_edge_index : half_edge_indices) {
                    const HalfEdge &half_edge = half_edges.at(half_edge_index);
                    const auto find_twin =
                        horizon_edge_indices.find(half_edge.twin_index);
                    if (find_twin == horizon_edge_indices.end()) {
                        horizon_edge_indices.insert(half_edge_index);
                    } else {
                        horizon_edge_indices.erase(find_twin);
                    }
                }
            }
        }
    }

    bool extend(const std::vector<Vector3D> &points, INDEX_T point_index) {
        std::vector<INDEX_T> visible_face_indices;
        std::set<INDEX_T> horizon_edge_indices;
        find_visible_faces(
            visible_face_indices,
            horizon_edge_indices,
            points,
            points.at(point_index)
        );
        for (const INDEX_T visible_face_index : visible_face_indices) {
            const Face visible_face = faces.at(visible_face_index);
            evict_face(visible_face_index);
            const auto half_edge_indices = get_half_edge_indices(visible_face);
            for (const INDEX_T half_edge_index : half_edge_indices) {
                if (horizon_edge_indices.find(half_edge_index) ==
                    horizon_edge_indices.end()) {
                    evict_half_edge(half_edge_index);
                }
            }
        }
        std::vector<INDEX_T> horizon_edge_list;
        horizon_edge_list.assign(
            horizon_edge_indices.begin(), horizon_edge_indices.end()
        );
        if (!cycle_sort_edges(horizon_edge_list)) { return false; }
        const INDEX_T num_horizon_edges =
            static_cast<INDEX_T>(horizon_edge_list.size());
        const INDEX_T num_new_half_edges =
            num_horizon_edges + num_horizon_edges;
        std::vector<INDEX_T> new_half_edge_indices;
        for (INDEX_T i = 0; i < num_new_half_edges; ++i) {
            new_half_edge_indices.push_back(add_empty_half_edge());
        }
        for (INDEX_T i = 0; i < num_horizon_edges; i++) {
            const INDEX_T ab_index = horizon_edge_list[i];
            HalfEdge &ab = half_edges.at(ab_index);
            const auto ab_indices = get_vertex_indices(ab);
            const INDEX_T new_face_index = add_empty_face();
            Face &new_face = faces.at(new_face_index);
            new_face.half_edge_index = ab_index;
            const INDEX_T ca_index = new_half_edge_indices[2 * i + 0];
            HalfEdge &ca = half_edges.at(ca_index);
            const INDEX_T bc_index = new_half_edge_indices[2 * i + 1];
            HalfEdge &bc = half_edges.at(bc_index);
            ab.next_index = bc_index;
            bc.next_index = ca_index;
            ca.next_index = ab_index;
            ab.face_index = new_face_index;
            bc.face_index = new_face_index;
            ca.face_index = new_face_index;
            bc.origin_index = ab_indices[1];
            ca.origin_index = point_index;
            bc.twin_index = new_half_edge_indices
                [(i + 1 == num_horizon_edges) ? 0 : 2 * i + 2];
            ca.twin_index = new_half_edge_indices
                [i == 0 ? (2 * num_horizon_edges - 1) : 2 * i - 1];
        }
        return true;
    }

}; // struct Polyhedron


static inline Polyhedron
construct_initial_tetrahedron(const std::vector<Vector3D> &points) {
    assert(points.size() >= 4);
    INDEX_T a_index = 0;
    INDEX_T b_index = 1;
    INDEX_T c_index = 2;
    INDEX_T d_index = 3;
    Vector3D a = points.at(0);
    Vector3D b = points.at(1);
    Vector3D c = points.at(2);
    Vector3D d = points.at(3);
    if (((b - a) ^ (c - a)) * (d - a) >= 0.0) {
        std::swap(c_index, d_index);
        std::swap(c, d);
    }
    Polyhedron result;
    result.faces.reserve(4);
    result.half_edges.reserve(12);
    result.faces.push_back({0});
    result.half_edges.push_back({a_index, 8, 0, 1});  // AB:  0 <-->  8
    result.half_edges.push_back({b_index, 11, 0, 2}); // BC:  1 <--> 11
    result.half_edges.push_back({c_index, 3, 0, 0});  // CA:  2 <-->  3
    result.faces.push_back({3});
    result.half_edges.push_back({a_index, 2, 1, 4});  // AC:  3 <-->  2
    result.half_edges.push_back({c_index, 10, 1, 5}); // CD:  4 <--> 10
    result.half_edges.push_back({d_index, 6, 1, 3});  // DA:  5 <-->  6
    result.faces.push_back({6});
    result.half_edges.push_back({a_index, 5, 2, 7}); //  AD:  6 <-->  5
    result.half_edges.push_back({d_index, 9, 2, 8}); //  DB:  7 <-->  9
    result.half_edges.push_back({b_index, 0, 2, 6}); //  BA:  8 <-->  0
    result.faces.push_back({9});
    result.half_edges.push_back({b_index, 7, 3, 10}); // BD:  9 <-->  7
    result.half_edges.push_back({d_index, 4, 3, 11}); // DC: 10 <-->  4
    result.half_edges.push_back({c_index, 1, 3, 9});  // CB: 11 <-->  1
    return result;
}


static inline Polyhedron convex_hull(const std::vector<Vector3D> &points) {
    const INDEX_T num_points = static_cast<INDEX_T>(points.size());
    if (num_points < 4) { return Polyhedron{}; }
    Polyhedron polyhedron = construct_initial_tetrahedron(points);
    for (INDEX_T point_index = 4; point_index < num_points; ++point_index) {
        if (!polyhedron.extend(points, point_index)) { return Polyhedron{}; }
    }
    return polyhedron;
}


void convex_hull(
    int *__restrict__ faces,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    const int num_faces = 2 * num_points - 4;
    if (num_points < 4) {
        for (int i = 0; i < num_faces; ++i) {
            faces[3 * i + 0] = -1;
            faces[3 * i + 1] = -1;
            faces[3 * i + 2] = -1;
        }
    }
    std::vector<Vector3D> points;
    points.reserve(static_cast<std::vector<Vector3D>::size_type>(num_points));
    for (int i = 0; i < num_points; ++i) {
        points.push_back(
            {static_cast<REAL_T>(points_x[i]),
             static_cast<REAL_T>(points_y[i]),
             static_cast<REAL_T>(points_z[i])}
        );
    }
    using vec_size_t = std::vector<Face>::size_type;
    const vec_size_t faces_size = static_cast<vec_size_t>(num_faces);
    const Polyhedron polyhedron = convex_hull(points);
    if (polyhedron.faces.size() == faces_size) {
        for (vec_size_t i = 0; i < faces_size; ++i) {
            const auto vertex_indices =
                polyhedron.get_vertex_indices(polyhedron.faces.at(i));
            faces[3 * i + 0] = static_cast<int>(vertex_indices[0]);
            faces[3 * i + 1] = static_cast<int>(vertex_indices[1]);
            faces[3 * i + 2] = static_cast<int>(vertex_indices[2]);
        }
    } else {
        for (int i = 0; i < num_faces; ++i) {
            faces[3 * i + 0] = -1;
            faces[3 * i + 1] = -1;
            faces[3 * i + 2] = -1;
        }
    }
}
