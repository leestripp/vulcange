#pragma once

#include "vge/math/matrix4x4.hpp"

namespace vulcanGE::math {

// ---------------------------------------------------------------------------
// Quaternion — unit quaternions for rotation (Plan §3.1: SIMD rotations).
// Layout: (x, y, z) vector part, w scalar part. Hamilton convention.
// ---------------------------------------------------------------------------
struct Quaternion {
    float x, y, z, w;

    constexpr Quaternion() noexcept : x(0.0F), y(0.0F), z(0.0F), w(1.0F) {}
    constexpr Quaternion(float x_in, float y_in, float z_in, float w_in) noexcept
        : x(x_in), y(y_in), z(z_in), w(w_in) {}

    [[nodiscard]] static constexpr Quaternion identity() noexcept { return {}; }

    [[nodiscard]] static Quaternion from_axis_angle(const Vector4& axis, float angle_rad) noexcept {
        const Vector4 n = axis.normalized3();
        const float half = angle_rad * 0.5F;
        const float s = std::sin(half);
        return {n.x * s, n.y * s, n.z * s, std::cos(half)};
    }

    // Hamilton product: applies rotation o first, then *this.
    [[nodiscard]] constexpr Quaternion operator*(const Quaternion& o) const noexcept {
        return {
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w,
            w * o.w - x * o.x - y * o.y - z * o.z,
        };
    }

    [[nodiscard]] constexpr float length_squared() const noexcept {
        return x * x + y * y + z * z + w * w;
    }
    [[nodiscard]] Quaternion normalized() const noexcept {
        const float len = std::sqrt(length_squared());
        return len > 0.0F ? Quaternion{x / len, y / len, z / len, w / len} : identity();
    }
    [[nodiscard]] constexpr Quaternion conjugate() const noexcept { return {-x, -y, -z, w}; }

    // Rotate a 3D vector: v' = q * (v,0) * q^-1 (optimized form).
    [[nodiscard]] Vector4 rotate(const Vector4& v) const noexcept {
        const Vector4 qv{x, y, z, 0.0F};
        const Vector4 t = qv.cross3(v) * 2.0F;
        return v + t * w + qv.cross3(t);
    }

    [[nodiscard]] constexpr Matrix4x4 to_matrix() const noexcept {
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float wx = w * x, wy = w * y, wz = w * z;
        Matrix4x4 m = Matrix4x4::identity();
        m.cols[0] = {1.0F - 2.0F * (yy + zz), 2.0F * (xy + wz), 2.0F * (xz - wy), 0.0F};
        m.cols[1] = {2.0F * (xy - wz), 1.0F - 2.0F * (xx + zz), 2.0F * (yz + wx), 0.0F};
        m.cols[2] = {2.0F * (xz + wy), 2.0F * (yz - wx), 1.0F - 2.0F * (xx + yy), 0.0F};
        return m;
    }

    // Normalized linear interpolation (shortest path).
    [[nodiscard]] constexpr Quaternion nlerp(const Quaternion& o, float t) const noexcept {
        const float sign = (x * o.x + y * o.y + z * o.z + w * o.w) < 0.0F ? -1.0F : 1.0F;
        return {
            x + (o.x * sign - x) * t,
            y + (o.y * sign - y) * t,
            z + (o.z * sign - z) * t,
            w + (o.w * sign - w) * t,
        };
    }
};

} // namespace vulcanGE::math
