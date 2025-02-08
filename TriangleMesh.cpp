#include "TriangleMesh.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <new>
#include <utility>
#include <vector>


struct Vector3D {
    double x;
    double y;
    double z;
};


constexpr Vector3D operator-(const Vector3D &v, const Vector3D &w) noexcept {
    return Vector3D{v.x - w.x, v.y - w.y, v.z - w.z};
}


constexpr double operator*(const Vector3D &v, const Vector3D &w) noexcept {
    return v.x * w.x + v.y * w.y + v.z * w.z;
}


constexpr Vector3D operator^(const Vector3D &v, const Vector3D &w) noexcept {
    return Vector3D{
        v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x
    };
}


static inline void find_initial_tetrahedron(
    int &a_index,
    int &b_index,
    int &c_index,
    int &d_index,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) noexcept {
    // Return early if there aren't enough vertices to form a tetrahedron.
    if (num_points < 4) {
        a_index = INVALID_INDEX;
        b_index = INVALID_INDEX;
        c_index = INVALID_INDEX;
        d_index = INVALID_INDEX;
        return;
    }
    // Find four vertices that lie furthest along the following directions:
    // A: (+1, +1, +1)
    // B: (+1, -1, -1)
    // C: (-1, +1, -1)
    // D: (-1, -1, +1)
    // These directions correspond to four opposite corners of a cube,
    // which form a regular tetrahedron.
    a_index = 0;
    b_index = 0;
    c_index = 0;
    d_index = 0;
    double a_best = points_x[0] + points_y[0] + points_z[0];
    double b_best = points_x[0] - points_y[0] - points_z[0];
    double c_best = points_y[0] - points_z[0] - points_x[0];
    double d_best = points_z[0] - points_x[0] - points_y[0];
    for (int i = 1; i < num_points; ++i) {
        const double x = points_x[i];
        const double y = points_y[i];
        const double z = points_z[i];
        const double a_score = x + y + z;
        const double b_score = x - y - z;
        const double c_score = y - z - x;
        const double d_score = z - x - y;
        if (a_score > a_best) {
            a_index = i;
            a_best = a_score;
        }
        if (b_score > b_best) {
            b_index = i;
            b_best = b_score;
        }
        if (c_score > c_best) {
            c_index = i;
            c_best = c_score;
        }
        if (d_score > d_best) {
            d_index = i;
            d_best = d_score;
        }
    }
    // Ensure that all four vertices found are distinct.
    if ((a_index == b_index) | (a_index == c_index) | (a_index == d_index) |
        (b_index == c_index) | (b_index == d_index) | (c_index == d_index)) {
        a_index = INVALID_INDEX;
        b_index = INVALID_INDEX;
        c_index = INVALID_INDEX;
        d_index = INVALID_INDEX;
    }
}


static inline void recursive_triangulate(
    std::vector<Triangle> &faces,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    const std::vector<int> &point_indices,
    const Triangle &triangle
) {
    const int a_index = triangle.ab.vertex_index;
    const int b_index = triangle.bc.vertex_index;
    const int c_index = triangle.ca.vertex_index;

    if (point_indices.size() == 3) {
        faces.push_back(triangle);
        return;
    }

    const Vector3D a{points_x[a_index], points_y[a_index], points_z[a_index]};
    const Vector3D b{points_x[b_index], points_y[b_index], points_z[b_index]};
    const Vector3D c{points_x[c_index], points_y[c_index], points_z[c_index]};
    const Vector3D normal = (b - a) ^ (c - a);
    int f_index = INVALID_INDEX;
    double f_best = 0.0;
    for (int i : point_indices) {
        const Vector3D p{points_x[i], points_y[i], points_z[i]};
        const double score = normal * p;
        if (score > f_best) {
            f_index = i;
            f_best = score;
        }
    }
    if ((f_index == a_index) | (f_index == b_index) | (f_index == c_index) |
        (f_index == INVALID_INDEX)) {
        return;
    }
    const Vector3D f{points_x[f_index], points_y[f_index], points_z[f_index]};

    const Vector3D fa = f ^ a;
    const Vector3D fb = f ^ b;
    const Vector3D fc = f ^ c;
    const Triangle abf{a_index, b_index, f_index};
    const Triangle bcf{b_index, c_index, f_index};
    const Triangle caf{c_index, a_index, f_index};

    std::vector<int> abf_points;
    std::vector<int> bcf_points;
    std::vector<int> caf_points;
    for (int i : point_indices) {
        const Vector3D p{points_x[i], points_y[i], points_z[i]};
        const bool is_f = (i == f_index);
        const bool in_fab = is_f | (i == a_index) | (i == b_index) |
                            ((p * fa >= 0.0) && (p * fb <= 0.0));
        const bool in_fbc = is_f | (i == b_index) | (i == c_index) |
                            ((p * fb >= 0.0) && (p * fc <= 0.0));
        const bool in_fca = is_f | (i == c_index) | (i == a_index) |
                            ((p * fc >= 0.0) && (p * fa <= 0.0));
        int count = in_fab + in_fbc + in_fca;
        const bool is_initial_vertex =
            ((i == a_index) | (i == b_index) | (i == c_index));
        const bool valid = ((count == 3) & is_f) |
                           ((count == 2) & (!is_f) & is_initial_vertex) |
                           ((count == 1) & (!is_f) & (!is_initial_vertex));
        if (!valid) { return; }
        if (in_fab) { abf_points.push_back(i); }
        if (in_fbc) { bcf_points.push_back(i); }
        if (in_fca) { caf_points.push_back(i); }
    }

    recursive_triangulate(faces, points_x, points_y, points_z, abf_points, abf);
    recursive_triangulate(faces, points_x, points_y, points_z, bcf_points, bcf);
    recursive_triangulate(faces, points_x, points_y, points_z, caf_points, caf);
}


static inline std::vector<Triangle> triangulate(
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    const int num_faces = 2 * num_points - 4;
    int a_index, b_index, c_index, d_index;
    find_initial_tetrahedron(
        a_index,
        b_index,
        c_index,
        d_index,
        points_x,
        points_y,
        points_z,
        num_points
    );
    if ((a_index == INVALID_INDEX) | (b_index == INVALID_INDEX) |
        (c_index == INVALID_INDEX) | (d_index == INVALID_INDEX)) {
        return {};
    }
    const Vector3D a{points_x[a_index], points_y[a_index], points_z[a_index]};
    const Vector3D b{points_x[b_index], points_y[b_index], points_z[b_index]};
    const Vector3D c{points_x[c_index], points_y[c_index], points_z[c_index]};
    const Vector3D d{points_x[d_index], points_y[d_index], points_z[d_index]};

    const Vector3D ab = a ^ b;
    const Vector3D ac = a ^ c;
    const Vector3D ad = a ^ d;
    const Vector3D bc = b ^ c;
    const Vector3D bd = b ^ d;
    const Vector3D cd = c ^ d;
    const Triangle abc{a_index, b_index, c_index};
    const Triangle acd{a_index, c_index, d_index};
    const Triangle adb{a_index, d_index, b_index};
    const Triangle bdc{b_index, d_index, c_index};

    std::vector<int> abc_points;
    std::vector<int> acd_points;
    std::vector<int> adb_points;
    std::vector<int> bdc_points;
    for (int i = 0; i < num_points; ++i) {
        const Vector3D p{points_x[i], points_y[i], points_z[i]};
        const bool in_abc =
            (i == a_index) | (i == b_index) | (i == c_index) |
            ((p * ab >= 0.0) && (p * bc >= 0.0) && (p * ac <= 0.0));
        const bool in_acd =
            (i == a_index) | (i == c_index) | (i == d_index) |
            ((p * ac >= 0.0) && (p * cd >= 0.0) && (p * ad <= 0.0));
        const bool in_adb =
            (i == a_index) | (i == d_index) | (i == b_index) |
            ((p * ad >= 0.0) && (p * bd <= 0.0) && (p * ab <= 0.0));
        const bool in_bdc =
            (i == b_index) | (i == d_index) | (i == c_index) |
            ((p * bd >= 0.0) && (p * cd <= 0.0) && (p * bc <= 0.0));
        const int count = in_abc + in_acd + in_adb + in_bdc;
        const bool is_initial_vertex =
            (i == a_index) | (i == b_index) | (i == c_index) | (i == d_index);
        const bool valid = ((count == 3) & is_initial_vertex) |
                           ((count == 1) & !is_initial_vertex);
        if (!valid) { return {}; }
        if (in_abc) { abc_points.push_back(i); }
        if (in_acd) { acd_points.push_back(i); }
        if (in_adb) { adb_points.push_back(i); }
        if (in_bdc) { bdc_points.push_back(i); }
    }

    std::vector<Triangle> faces;
    recursive_triangulate(faces, points_x, points_y, points_z, abc_points, abc);
    recursive_triangulate(faces, points_x, points_y, points_z, acd_points, acd);
    recursive_triangulate(faces, points_x, points_y, points_z, adb_points, adb);
    recursive_triangulate(faces, points_x, points_y, points_z, bdc_points, bdc);

    using vec_size_t = std::vector<Triangle>::size_type;
    const vec_size_t expected_size =
        2 * static_cast<vec_size_t>(num_points) - 4;
    if (faces.size() == expected_size) {
        return faces;
    } else {
        return {};
    }
}


static inline HalfEdge *
get_half_edge(Triangle *faces, int edge_index) noexcept {
    assert(faces);
    assert(edge_index >= 0);
    // return reinterpret_cast<HalfEdge *>(faces) + edge_index;
    switch (edge_index % 3) {
        case 0: return &faces[edge_index / 3].ab;
        case 1: return &faces[edge_index / 3].bc;
        case 2: return &faces[edge_index / 3].ca;
    }
    assert(false);
}


static inline bool compute_twins(Triangle *faces, int num_faces) {
    std::map<std::pair<int, int>, int> unmatched_edges;
    for (int i = 0; i < num_faces; ++i) {
        const Triangle face = faces[i];
        const int a_index = face.ab.vertex_index;
        const int b_index = face.bc.vertex_index;
        const int c_index = face.ca.vertex_index;
        const int ab_index = 3 * i + 0;
        assert(get_half_edge(faces, ab_index)->vertex_index == a_index);
        assert(get_half_edge(faces, ab_index)->twin_index == INVALID_INDEX);
        const auto find_ab =
            unmatched_edges.find(std::make_pair(a_index, b_index));
        if (find_ab != unmatched_edges.end()) {
            const int ba_index = find_ab->second;
            get_half_edge(faces, ab_index)->twin_index = ba_index;
            get_half_edge(faces, ba_index)->twin_index = ab_index;
            unmatched_edges.erase(find_ab);
        } else {
            unmatched_edges[std::make_pair(b_index, a_index)] = ab_index;
        }
        const int bc_index = 3 * i + 1;
        assert(get_half_edge(faces, bc_index)->vertex_index == b_index);
        assert(get_half_edge(faces, bc_index)->twin_index == INVALID_INDEX);
        const auto find_bc =
            unmatched_edges.find(std::make_pair(b_index, c_index));
        if (find_bc != unmatched_edges.end()) {
            const int cb_index = find_bc->second;
            get_half_edge(faces, bc_index)->twin_index = cb_index;
            get_half_edge(faces, cb_index)->twin_index = bc_index;
            unmatched_edges.erase(find_bc);
        } else {
            unmatched_edges[std::make_pair(c_index, b_index)] = bc_index;
        }
        const int ca_index = 3 * i + 2;
        assert(get_half_edge(faces, ca_index)->vertex_index == c_index);
        assert(get_half_edge(faces, ca_index)->twin_index == INVALID_INDEX);
        const auto find_ca =
            unmatched_edges.find(std::make_pair(c_index, a_index));
        if (find_ca != unmatched_edges.end()) {
            const int ac_index = find_ca->second;
            get_half_edge(faces, ca_index)->twin_index = ac_index;
            get_half_edge(faces, ac_index)->twin_index = ca_index;
            unmatched_edges.erase(find_ca);
        } else {
            unmatched_edges[std::make_pair(a_index, c_index)] = ca_index;
        }
    }
    assert(unmatched_edges.empty());
    for (int i = 0; i <) return unmatched_edges.empty();
}


TriangleMesh::TriangleMesh(
    const double *__restrict__ vertices_x,
    const double *__restrict__ vertices_y,
    const double *__restrict__ vertices_z,
    int num_vertices
) {
    num_faces = 2 * num_vertices - 4;
    faces = new (std::nothrow) Triangle[num_faces];
    assert(faces);
    if (faces) {
        const std::vector<Triangle> triangles =
            triangulate(vertices_x, vertices_y, vertices_z, num_vertices);
        assert(triangles.size() == static_cast<std::size_t>(num_faces));
        if (triangles.size() == static_cast<std::size_t>(num_faces)) {
            for (int i = 0; i < num_faces; ++i) { faces[i] = triangles[i]; }
            const bool success = compute_twins(faces, num_faces);
            assert(success);
        }
    }
    assert(is_allocated());
}


TriangleMesh::~TriangleMesh() noexcept {
    if (faces) { delete[] faces; }
}


bool TriangleMesh::is_allocated() const noexcept {
    return static_cast<bool>(faces);
}


int next_half_edge_index(int edge_index) noexcept {
    assert(edge_index >= 0);
    const int face_index = edge_index / 3;
    switch (edge_index % 3) {
        case 0: return 3 * face_index + 1;
        case 1: return 3 * face_index + 2;
        case 2: return 3 * face_index + 0;
    }
    assert(false);
}


bool TriangleMesh::flip_edges(
    const double *__restrict__ x,
    const double *__restrict__ y,
    const double *__restrict__ z
) noexcept {

    assert(is_allocated());
    for (int i = 0; i < num_faces; ++i) {

        const int ab_index = 3 * i + 0;
        const int bc_index = 3 * i + 1;
        const int ca_index = 3 * i + 2;
        HalfEdge *ab = get_half_edge(faces, ab_index);
        HalfEdge *bc = get_half_edge(faces, bc_index);
        HalfEdge *ca = get_half_edge(faces, ca_index);
        assert(ab->vertex_index >= 0);
        assert(ab->twin_index >= 0);
        assert(bc->vertex_index >= 0);
        assert(bc->twin_index >= 0);
        assert(ca->vertex_index >= 0);
        assert(ca->twin_index >= 0);
        const int a_index = ab->vertex_index;
        const int b_index = bc->vertex_index;
        const int c_index = ca->vertex_index;
        assert(a_index >= 0);
        assert(b_index >= 0);
        assert(c_index >= 0);

        const int ba_index = ab->twin_index;
        const int ad_index = next_half_edge_index(ba_index);
        const int db_index = next_half_edge_index(ad_index);
        assert(ba_index >= 0);
        assert(ad_index >= 0);
        assert(db_index >= 0);
        HalfEdge *ba = get_half_edge(faces, ba_index);
        HalfEdge *ad = get_half_edge(faces, ad_index);
        HalfEdge *db = get_half_edge(faces, db_index);
        assert(ba->vertex_index >= 0);
        assert(ba->twin_index >= 0);
        assert(ad->vertex_index >= 0);
        assert(ad->twin_index >= 0);
        assert(db->vertex_index >= 0);
        assert(db->twin_index >= 0);
        assert(ba->vertex_index == b_index);
        assert(ad->vertex_index == a_index);
        assert(ba->twin_index == ab_index);

        const int d_index = db->vertex_index;
        assert(d_index >= 0);

        const Vector3D a{x[a_index], y[a_index], z[a_index]};
        const Vector3D b{x[b_index], y[b_index], z[b_index]};
        const Vector3D c{x[c_index], y[c_index], z[c_index]};
        const Vector3D d{x[d_index], y[d_index], z[d_index]};

        // if ((c - d) * (c - d) < (a - b) * (a - b)) {
        //     const int ad_twin_index = ad.twin_index;
        //     const int bc_twin_index = bc.twin_index;
        //     // ab becomes ad
        //     ab.twin_index = ad_twin_index;
        //     // bc becomes dc
        //     bc.vertex_index = d_index;
        //     bc.twin_index = ad_index;
        //     // ba becomes bc
        //     ba.twin_index = bc_twin_index;
        //     // ad becomes cd
        //     ad.vertex_index = c_index;
        //     ad.twin_index = bc_index;
        // }
    }
}


Triangle TriangleMesh::get_face(int index) const noexcept {
    return faces[index];
}
