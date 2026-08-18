#pragma once

#include "vge/math/vector4.hpp"

namespace vulcanGE::math {

// ---------------------------------------------------------------------------
// Matrix4x4 — column-major, cache-line aligned (Plan §3.4).
// Columns are stored contiguously: element (row, col) = cols[col][row].
// All projections are Vulkan-native: Y-down clip space, reverse-Z depth with
// depth 1.0 at the near plane and 0.0 at the far plane / infinity (Plan §3.3).
// ---------------------------------------------------------------------------
struct alignas(64) Matrix4x4 {
    Vector4 cols[4];

    constexpr Matrix4x4() noexcept = default;

    [[nodiscard]] static constexpr Matrix4x4 identity() noexcept {
        Matrix4x4 m{};
        m.cols[0].x = 1.0F;
        m.cols[1].y = 1.0F;
        m.cols[2].z = 1.0F;
        m.cols[3].w = 1.0F;
        return m;
    }

    // -- Reverse-Z perspective, infinite far plane (Plan §3.3, default) ------
    // Near plane -> depth 1.0, infinity -> depth 0.0.
    [[nodiscard]] static constexpr Matrix4x4 perspective_vulkan(
        float fov_y_rad, float aspect, float z_near) noexcept {
        Matrix4x4 m{};
        const float focal = 1.0F / std::tan(fov_y_rad * 0.5F);
        m.cols[0].x = focal / aspect;
        m.cols[1].y = -focal; // Y-flip: Vulkan clip space is Y-down
        m.cols[2].z = 0.0F;   // reverse-Z infinite far plane
        m.cols[2].w = -1.0F;
        m.cols[3].z = z_near;
        return m;
    }

    // -- Reverse-Z perspective, finite far plane (tools/validation only) -----
    // Near plane -> depth 1.0, far plane -> depth 0.0.
    [[nodiscard]] static constexpr Matrix4x4 perspective_vulkan_finite(
        float fov_y_rad, float aspect, float z_near, float z_far) noexcept {
        Matrix4x4 m{};
        const float focal = 1.0F / std::tan(fov_y_rad * 0.5F);
        const float range = z_far - z_near;
        m.cols[0].x = focal / aspect;
        m.cols[1].y = -focal;
        m.cols[2].z = z_near / range;
        m.cols[2].w = -1.0F;
        m.cols[3].z = (z_near * z_far) / range;
        return m;
    }

    // -- Reverse-Z orthographic (shadow cascades, tools, 2D UI) -------------
    // View volume [l,r] x [b,t] x [z_near,z_far]; near -> 1.0, far -> 0.0.
    [[nodiscard]] static constexpr Matrix4x4 ortho_vulkan(
        float left, float right, float bottom, float top,
        float z_near, float z_far) noexcept {
        Matrix4x4 m{};
        const float inv_w = 1.0F / (right - left);
        const float inv_h = 1.0F / (bottom - top);
        const float inv_d = 1.0F / (z_far - z_near);
        m.cols[0].x = 2.0F * inv_w;
        m.cols[1].y = 2.0F * inv_h; // Y-flip baked in
        m.cols[2].z = inv_d;
        m.cols[3].x = -(right + left) * inv_w;
        m.cols[3].y = -(bottom + top) * inv_h;
        m.cols[3].z = z_far * inv_d;
        m.cols[3].w = 1.0F;
        return m;
    }

    // -- Right-handed view matrix; camera looks down -Z (Plan §3.2) ---------
    [[nodiscard]] static constexpr Matrix4x4 look_at_rh(
        const Vector4& eye, const Vector4& target, const Vector4& up) noexcept {
        const Vector4 f = (target - eye).normalized3();
        const Vector4 s = f.cross3(up).normalized3();
        const Vector4 u = s.cross3(f);
        Matrix4x4 m = identity();
        m.cols[0] = {s.x, u.x, -f.x, 0.0F};
        m.cols[1] = {s.y, u.y, -f.y, 0.0F};
        m.cols[2] = {s.z, u.z, -f.z, 0.0F};
        m.cols[3] = {-s.dot3(eye), -u.dot3(eye), f.dot3(eye), 1.0F};
        return m;
    }

    // -- Scalar constexpr multiply (reference implementation) ----------------
    [[nodiscard]] constexpr Matrix4x4 operator*(const Matrix4x4& o) const noexcept {
        Matrix4x4 r{};
        for (std::size_t c = 0; c < 4; ++c) {
            for (std::size_t row = 0; row < 4; ++row) {
                float sum = 0.0F;
                for (std::size_t k = 0; k < 4; ++k) {
                    sum += cols[k][row] * o.cols[c][k];
                }
                r.cols[c][row] = sum;
            }
        }
        return r;
    }

    // Full 4D transform.
    [[nodiscard]] constexpr Vector4 transform(const Vector4& v) const noexcept {
        return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z + cols[3] * v.w;
    }
    // Position transform (w = 1), perspective divide applied.
    [[nodiscard]] constexpr Vector4 transform_point(const Vector4& p) const noexcept {
        const Vector4 r = transform({p.x, p.y, p.z, 1.0F});
        return r.w != 0.0F ? r / r.w : r;
    }
    // Direction transform (w = 0, no translation, no divide).
    [[nodiscard]] constexpr Vector4 transform_vector(const Vector4& v) const noexcept {
        return transform({v.x, v.y, v.z, 0.0F});
    }

    // -- General 4x4 inverse (adjugate / determinant) ------------------------
    // Canonical gluInvertMatrix formulation; input/output are column-major
    // (our layout). Needed for world-space reconstruction (inv view-proj) and
    // shadow cascade frustum corners. Runtime-only; not on the hot path.
    [[nodiscard]] constexpr Matrix4x4 inverse() const noexcept {
        const float* m = &cols[0].x; // column-major
        float inv[16];
        inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] +
                 m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
        inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] -
                 m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
        inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] +
                 m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
        inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] -
                  m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
        inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] -
                 m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
        inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] +
                 m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
        inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] -
                 m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
        inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] +
                  m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
        inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] +
                 m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
        inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] -
                 m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
        inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] +
                  m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
        inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] -
                  m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
        inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] -
                 m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
        inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] +
                 m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
        inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] -
                  m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
        inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] +
                  m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
        const float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
        const float inv_det = 1.0F / det;
        Matrix4x4 r{};
        float* out = &r.cols[0].x;
        for (std::size_t i = 0; i < 16; ++i) {
            out[i] = inv[i] * inv_det;
        }
        return r;
    }

    [[nodiscard]] constexpr bool approx_equal(const Matrix4x4& o, float eps = 1e-5F) const noexcept {
        return cols[0].approx_equal(o.cols[0], eps) && cols[1].approx_equal(o.cols[1], eps) &&
               cols[2].approx_equal(o.cols[2], eps) && cols[3].approx_equal(o.cols[3], eps);
    }
};

// ---------------------------------------------------------------------------
// SIMD-accelerated multiply (SSE 4-wide). Falls back to the scalar constexpr
// path on targets without SSE (ARM NEON path arrives with the ARM64 port).
// Runtime-only: intrinsics are not constexpr — the scalar operator* remains
// the canonical reference for tests (Plan §16.4 SIMD-vs-scalar validation).
// ---------------------------------------------------------------------------
[[nodiscard]] inline Matrix4x4 multiply_simd(const Matrix4x4& a, const Matrix4x4& b) noexcept {
#if VGE_MATH_SIMD_SSE
    Matrix4x4 r{};
    for (std::size_t c = 0; c < 4; ++c) {
        const float* bc = &b.cols[c].x;
        __m128 acc = _mm_mul_ps(_mm_set1_ps(bc[0]), _mm_load_ps(&a.cols[0].x));
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(bc[1]), _mm_load_ps(&a.cols[1].x)));
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(bc[2]), _mm_load_ps(&a.cols[2].x)));
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(bc[3]), _mm_load_ps(&a.cols[3].x)));
        _mm_store_ps(&r.cols[c].x, acc);
    }
    return r;
#else
    return a * b;
#endif
}

} // namespace vulcanGE::math
