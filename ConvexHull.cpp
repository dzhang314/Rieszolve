#include "ConvexHull.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>


struct Vector3D {
    double x;
    double y;
    double z;
};


struct FaceIndices3D {
    int i;
    int j;
    int k;
};


static inline Vector3D operator-(const Vector3D &v, const Vector3D &w) {
    return Vector3D{v.x - w.x, v.y - w.y, v.z - w.z};
}


static inline double dot(const Vector3D &v, const Vector3D &w) {
    return v.x * w.x + v.y * w.y + v.z * w.z;
}


static inline Vector3D cross(const Vector3D &v, const Vector3D &w) {
    return Vector3D{
        v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x
    };
}


template <typename T>
static inline bool contains(const std::vector<T> &items, const T &item) {
    return std::find(items.begin(), items.end(), item) != items.end();
}


static void recursive_triangulate(
    std::vector<FaceIndices3D> &faces,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    const std::vector<int> &point_indices,
    const FaceIndices3D &face_indices
) {
    const int a_index = face_indices.i;
    const int b_index = face_indices.j;
    const int c_index = face_indices.k;
    assert(contains(point_indices, a_index));
    assert(contains(point_indices, b_index));
    assert(contains(point_indices, c_index));

    assert(point_indices.size() >= 3);
    if (point_indices.size() == 3) {
        faces.push_back(face_indices);
        return;
    }

    const Vector3D a{points_x[a_index], points_y[a_index], points_z[a_index]};
    const Vector3D b{points_x[b_index], points_y[b_index], points_z[b_index]};
    const Vector3D c{points_x[c_index], points_y[c_index], points_z[c_index]};
    const Vector3D normal = cross(b - a, c - a);
    int f_index = -1;
    double f_best = 0.0;
    for (int i : point_indices) {
        const Vector3D p{points_x[i], points_y[i], points_z[i]};
        const double score = dot(normal, p);
        if (score > f_best) {
            f_index = i;
            f_best = score;
        }
    }
    if ((f_index == a_index) | (f_index == b_index) | (f_index == c_index) |
        (f_index == -1)) {
        return;
    }
    const Vector3D f{points_x[f_index], points_y[f_index], points_z[f_index]};

    const Vector3D fa = cross(f, a);
    const Vector3D fb = cross(f, b);
    const Vector3D fc = cross(f, c);
    const FaceIndices3D abf_indices{a_index, b_index, f_index};
    const FaceIndices3D bcf_indices{b_index, c_index, f_index};
    const FaceIndices3D caf_indices{c_index, a_index, f_index};

    std::vector<int> abf_points;
    std::vector<int> bcf_points;
    std::vector<int> caf_points;
    for (int i : point_indices) {
        const Vector3D p{points_x[i], points_y[i], points_z[i]};
        const bool in_fab = (i == f_index) | (i == a_index) | (i == b_index) |
                            ((dot(p, fa) >= 0.0) & (dot(p, fb) <= 0.0));
        const bool in_fbc = (i == f_index) | (i == b_index) | (i == c_index) |
                            ((dot(p, fb) >= 0.0) & (dot(p, fc) <= 0.0));
        const bool in_fca = (i == f_index) | (i == c_index) | (i == a_index) |
                            ((dot(p, fc) >= 0.0) & (dot(p, fa) <= 0.0));
        if (in_fab) { abf_points.push_back(i); }
        if (in_fbc) { bcf_points.push_back(i); }
        if (in_fca) { caf_points.push_back(i); }
        // assert(in_fab | in_fbc | in_fca);
    }

    recursive_triangulate(
        faces, points_x, points_y, points_z, abf_points, abf_indices
    );
    recursive_triangulate(
        faces, points_x, points_y, points_z, bcf_points, bcf_indices
    );
    recursive_triangulate(
        faces, points_x, points_y, points_z, caf_points, caf_indices
    );
}


void triangulate(
    int *__restrict__ faces,
    const double *__restrict__ points_x,
    const double *__restrict__ points_y,
    const double *__restrict__ points_z,
    int num_points
) {
    int a_index = 0;
    int b_index = 0;
    int c_index = 0;
    int d_index = 0;
    double a_best = points_x[0] + points_y[0] + points_z[0];
    double b_best = points_x[0] - points_y[0] - points_z[0];
    double c_best = points_y[0] - points_z[0] - points_x[0];
    double d_best = points_z[0] - points_x[0] - points_y[0];
    for (int i = 1; i < num_points; ++i) {
        const double a_score = points_x[i] + points_y[i] + points_z[i];
        const double b_score = points_x[i] - points_y[i] - points_z[i];
        const double c_score = points_y[i] - points_z[i] - points_x[i];
        const double d_score = points_z[i] - points_x[i] - points_y[i];
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
    const int num_faces = 2 * num_points - 4;
    if ((a_index == b_index) | (a_index == c_index) | (a_index == d_index) |
        (b_index == c_index) | (b_index == d_index) | (c_index == d_index)) {
        for (int i = 0; i < 3 * num_faces; ++i) { faces[i] = -1; }
        return;
    }
    const Vector3D a{points_x[a_index], points_y[a_index], points_z[a_index]};
    const Vector3D b{points_x[b_index], points_y[b_index], points_z[b_index]};
    const Vector3D c{points_x[c_index], points_y[c_index], points_z[c_index]};
    const Vector3D d{points_x[d_index], points_y[d_index], points_z[d_index]};

    const Vector3D ab = cross(a, b);
    const Vector3D ac = cross(a, c);
    const Vector3D ad = cross(a, d);
    const Vector3D bc = cross(b, c);
    const Vector3D bd = cross(b, d);
    const Vector3D cd = cross(c, d);
    const FaceIndices3D abc_indices{a_index, b_index, c_index};
    const FaceIndices3D acd_indices{a_index, c_index, d_index};
    const FaceIndices3D adb_indices{a_index, d_index, b_index};
    const FaceIndices3D bdc_indices{b_index, d_index, c_index};

    std::vector<int> abc_points;
    std::vector<int> acd_points;
    std::vector<int> adb_points;
    std::vector<int> bdc_points;
    for (int i = 0; i < num_points; ++i) {
        const Vector3D p{points_x[i], points_y[i], points_z[i]};
        const bool in_abc =
            (i == a_index) | (i == b_index) | (i == c_index) |
            ((dot(p, ab) >= 0.0) & (dot(p, bc) >= 0.0) & (dot(p, ac) <= 0.0));
        const bool in_acd =
            (i == a_index) | (i == c_index) | (i == d_index) |
            ((dot(p, ac) >= 0.0) & (dot(p, cd) >= 0.0) & (dot(p, ad) <= 0.0));
        const bool in_adb =
            (i == a_index) | (i == d_index) | (i == b_index) |
            ((dot(p, ad) >= 0.0) & (dot(p, bd) <= 0.0) & (dot(p, ab) <= 0.0));
        const bool in_bdc =
            (i == b_index) | (i == d_index) | (i == c_index) |
            ((dot(p, bd) >= 0.0) & (dot(p, cd) <= 0.0) & (dot(p, bc) <= 0.0));
        if (in_abc) { abc_points.push_back(i); }
        if (in_acd) { acd_points.push_back(i); }
        if (in_adb) { adb_points.push_back(i); }
        if (in_bdc) { bdc_points.push_back(i); }
        // assert(in_abc | in_acd | in_adb | in_bdc);
    }

    std::vector<FaceIndices3D> result;
    recursive_triangulate(
        result, points_x, points_y, points_z, abc_points, abc_indices
    );
    recursive_triangulate(
        result, points_x, points_y, points_z, acd_points, acd_indices
    );
    recursive_triangulate(
        result, points_x, points_y, points_z, adb_points, adb_indices
    );
    recursive_triangulate(
        result, points_x, points_y, points_z, bdc_points, bdc_indices
    );

    using vec_size_t = std::vector<FaceIndices3D>::size_type;
    const vec_size_t expected_size =
        static_cast<vec_size_t>(2 * num_points - 4);

    if (result.size() == expected_size) {
        int k = 0;
        for (const FaceIndices3D &face : result) {
            faces[k++] = face.i;
            faces[k++] = face.j;
            faces[k++] = face.k;
        }
    } else {
        for (int i = 0; i < 3 * num_faces; ++i) { faces[i] = -1; }
    }
}
