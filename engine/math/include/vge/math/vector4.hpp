#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#if defined(__SSE__)
#    include <immintrin.h>
#    define VGE_MATH_SIMD_SSE 1
#else
#    define VGE_MATH_SIMD_SSE 0
#endif

namespace vulcanGE::math {

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

// ---------------------------------------------------------------------------
// Vector4 — 16-byte aligned SIMD-friendly 4D vector (Plan §3.4).
// Scalar constexpr core; SIMD acceleration lives in free functions where
// runtime-only intrinsics are allowed.
// ---------------------------------------------------------------------------
struct alignas(16) Vector4 {
    float x, y, z, w;

    constexpr Vector4() noexcept : x(0.0F), y(0.0F), z(0.0F), w(0.0F) {}
    constexpr Vector4(float x_in, float y_in, float z_in, float w_in) noexcept
        : x(x_in), y(y_in), z(z_in), w(w_in) {}

    [[nodiscard]] static constexpr Vector4 splat(float v) noexcept { return {v, v, v, v}; }

    // Array access: 0=x, 1=y, 2=z, 3=w
    [[nodiscard]] constexpr float operator[](std::size_t i) const noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return (&x)[i]; // contiguous by standard layout
    }
    [[nodiscard]] constexpr float& operator[](std::size_t i) noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return (&x)[i];
    }

    [[nodiscard]] constexpr Vector4 operator+(const Vector4& o) const noexcept {
        return {x + o.x, y + o.y, z + o.z, w + o.w};
    }
    [[nodiscard]] constexpr Vector4 operator-(const Vector4& o) const noexcept {
        return {x - o.x, y - o.y, z - o.z, w - o.w};
    }
    [[nodiscard]] constexpr Vector4 operator*(float s) const noexcept {
        return {x * s, y * s, z * s, w * s};
    }
    [[nodiscard]] constexpr Vector4 operator/(float s) const noexcept {
        return {x / s, y / s, z / s, w / s};
    }
    [[nodiscard]] constexpr Vector4 operator-() const noexcept { return {-x, -y, -z, -w}; }

    [[nodiscard]] constexpr float dot(const Vector4& o) const noexcept {
        return x * o.x + y * o.y + z * o.z + w * o.w;
    }
    [[nodiscard]] constexpr float dot3(const Vector4& o) const noexcept {
        return x * o.x + y * o.y + z * o.z;
    }
    [[nodiscard]] constexpr Vector4 cross3(const Vector4& o) const noexcept {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x, 0.0F};
    }

    [[nodiscard]] constexpr float length3_squared() const noexcept { return dot3(*this); }
    [[nodiscard]] float length3() const noexcept { return std::sqrt(length3_squared()); }
    [[nodiscard]] Vector4 normalized3() const noexcept {
        const float len = length3();
        return len > 0.0F ? (*this) / len : Vector4{};
    }

    [[nodiscard]] constexpr bool approx_equal(const Vector4& o, float eps = 1e-5F) const noexcept {
        return std::fabs(x - o.x) <= eps && std::fabs(y - o.y) <= eps &&
               std::fabs(z - o.z) <= eps && std::fabs(w - o.w) <= eps;
    }
};

[[nodiscard]] constexpr Vector4 operator*(float s, const Vector4& v) noexcept { return v * s; }

} // namespace vulcanGE::math
