#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <random>

#include <vge/math/math.hpp>

using namespace vulcanGE::math;

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

Matrix4x4 random_matrix(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-10.0F, 10.0F);
    Matrix4x4 m{};
    for (std::size_t c = 0; c < 4; ++c) {
        for (std::size_t r = 0; r < 4; ++r) {
            m.cols[c][r] = dist(rng);
        }
    }
    return m;
}

} // namespace

TEST_CASE("Vector4: layout and alignment invariants") {
    static_assert(alignof(Vector4) == 16);
    static_assert(sizeof(Vector4) == 16);
    static_assert(alignof(Matrix4x4) == 64);
    static_assert(sizeof(Matrix4x4) == 64);
    CHECK(true); // compile-time invariants held
}

TEST_CASE("Vector4: dot / cross / normalize") {
    const Vector4 a{1, 2, 3, 0};
    const Vector4 b{4, -5, 6, 0};
    CHECK(a.dot3(b) == doctest::Approx(4 - 10 + 18));
    CHECK(a.cross3(b).approx_equal({27, 6, -13, 0}));
    CHECK(Vector4{3, 4, 0, 0}.length3() == doctest::Approx(5.0));
    const Vector4 n = Vector4{0, 3, 4, 0}.normalized3();
    CHECK(n.length3() == doctest::Approx(1.0));
    CHECK(n.approx_equal({0, 0.6F, 0.8F, 0}));
}

TEST_CASE("Matrix4x4: identity is a multiplicative unit") {
    std::mt19937 rng(42);
    const Matrix4x4 m = random_matrix(rng);
    CHECK((Matrix4x4::identity() * m).approx_equal(m));
    CHECK((m * Matrix4x4::identity()).approx_equal(m));
}

TEST_CASE("Matrix4x4: reverse-Z infinite perspective (Plan 3.3)") {
    constexpr float fov = kPi / 3.0F; // 60 deg
    constexpr float aspect = 16.0F / 9.0F;
    constexpr float zn = 0.1F;
    const Matrix4x4 p = Matrix4x4::perspective_vulkan(fov, aspect, zn);

    SUBCASE("near plane maps to depth 1.0") {
        const Vector4 ndc = p.transform_point({0, 0, -zn, 0});
        CHECK(ndc.z == doctest::Approx(1.0));
    }
    SUBCASE("far distances approach depth 0.0 but never reach it") {
        const Vector4 ndc_far = p.transform_point({0, 0, -10000.0F, 0});
        CHECK(ndc_far.z == doctest::Approx(0.0).epsilon(1e-4));
        CHECK(ndc_far.z > 0.0F);
    }
    SUBCASE("depth is monotonically decreasing with distance (reversed)") {
        const float d1 = p.transform_point({0, 0, -1.0F, 0}).z;
        const float d2 = p.transform_point({0, 0, -10.0F, 0}).z;
        const float d3 = p.transform_point({0, 0, -100.0F, 0}).z;
        CHECK(d1 > d2);
        CHECK(d2 > d3);
    }
    SUBCASE("Y axis is flipped for Vulkan clip space") {
        // A point above the view axis must land at negative NDC Y (= screen top).
        const Vector4 ndc = p.transform_point({0, 1, -1, 0});
        CHECK(ndc.y < 0.0F);
    }
    SUBCASE("aspect ratio scales X correctly") {
        const Vector4 ndc = p.transform_point({1, 0, -1, 0});
        const float expected = (1.0F / std::tan(fov * 0.5F)) / aspect;
        CHECK(ndc.x == doctest::Approx(expected));
    }
}

TEST_CASE("Matrix4x4: reverse-Z finite perspective bounds") {
    const Matrix4x4 p = Matrix4x4::perspective_vulkan_finite(kPi / 3.0F, 1.0F, 0.5F, 500.0F);
    CHECK(p.transform_point({0, 0, -0.5F, 0}).z == doctest::Approx(1.0));
    CHECK(p.transform_point({0, 0, -500.0F, 0}).z == doctest::Approx(0.0));
}

TEST_CASE("Matrix4x4: reverse-Z orthographic bounds") {
    const Matrix4x4 o = Matrix4x4::ortho_vulkan(-10, 10, -5, 5, 0.1F, 100.0F);
    CHECK(o.transform_point({0, 0, -0.1F, 0}).z == doctest::Approx(1.0));
    CHECK(o.transform_point({0, 0, -100.0F, 0}).z == doctest::Approx(0.0));
    CHECK(o.transform_point({-10, 0, -1, 0}).x == doctest::Approx(-1.0));
    CHECK(o.transform_point({10, 0, -1, 0}).x == doctest::Approx(1.0));
    CHECK(o.transform_point({0, 5, -1, 0}).y == doctest::Approx(-1.0)); // top -> NDC -1 (Y down)
    CHECK(o.transform_point({0, -5, -1, 0}).y == doctest::Approx(1.0));
}

TEST_CASE("Matrix4x4: look_at_rh identity at origin looking down -Z") {
    const Matrix4x4 v = Matrix4x4::look_at_rh({0, 0, 0, 0}, {0, 0, -1, 0}, {0, 1, 0, 0});
    CHECK(v.approx_equal(Matrix4x4::identity()));
}

TEST_CASE("Matrix4x4: look_at_rh translation moves world into view space") {
    const Matrix4x4 v = Matrix4x4::look_at_rh({0, 0, 5, 0}, {0, 0, 4, 0}, {0, 1, 0, 0});
    const Vector4 view = v.transform_point({0, 0, 5, 0});
    CHECK(view.approx_equal({0, 0, 0, 1}));
}

TEST_CASE("Matrix4x4: SIMD multiply matches scalar reference (Plan 16.4)") {
    std::mt19937 rng(1337);
    for (int i = 0; i < 256; ++i) {
        const Matrix4x4 a = random_matrix(rng);
        const Matrix4x4 b = random_matrix(rng);
        const Matrix4x4 scalar = a * b;
        const Matrix4x4 simd = multiply_simd(a, b);
        CHECK(simd.approx_equal(scalar, 1e-3F));
    }
}

TEST_CASE("Quaternion: axis-angle rotation round trip") {
    const Quaternion q = Quaternion::from_axis_angle({0, 1, 0, 0}, kPi / 2.0F);
    const Vector4 rotated = q.rotate({1, 0, 0, 0});
    CHECK(rotated.approx_equal({0, 0, -1, 0}, 1e-5F)); // +90 deg about Y: +X -> -Z (RH)
    CHECK(q.length_squared() == doctest::Approx(1.0));
}

TEST_CASE("Quaternion: to_matrix matches rotate()") {
    const Quaternion q =
        Quaternion::from_axis_angle({1, 2, 3, 0}, 1.1F).normalized();
    const Matrix4x4 m = q.to_matrix();
    const Vector4 v{0.3F, -0.7F, 1.2F, 0.0F};
    CHECK(m.transform_vector(v).approx_equal(q.rotate(v), 1e-4F));
}

TEST_CASE("Quaternion: nlerp midpoint interpolation") {
    const Quaternion a = Quaternion::identity();
    const Quaternion b = Quaternion::from_axis_angle({0, 0, 1, 0}, 2.0F * kPi / 3.0F); // 120 deg
    const Quaternion mid = a.nlerp(b, 0.5F).normalized();
    const Quaternion expected = Quaternion::from_axis_angle({0, 0, 1, 0}, kPi / 3.0F); // 60 deg
    CHECK(mid.rotate({1, 0, 0, 0}).approx_equal(expected.rotate({1, 0, 0, 0}), 1e-4F));
}

TEST_CASE("Quaternion: nlerp takes shortest path across antipodal quaternions") {
    // -q encodes the same rotation as q; interpolation must not go the long way.
    const Quaternion a = Quaternion::identity();
    const Quaternion b_pos = Quaternion::from_axis_angle({0, 1, 0, 0}, 0.5F);
    const Quaternion b_neg{-b_pos.x, -b_pos.y, -b_pos.z, -b_pos.w}; // same rotation, negated
    const Quaternion mid = a.nlerp(b_neg, 0.5F).normalized();
    const Quaternion expected = Quaternion::from_axis_angle({0, 1, 0, 0}, 0.25F);
    const Vector4 v{0.4F, 0.2F, -0.9F, 0.0F};
    CHECK(mid.rotate(v).approx_equal(expected.rotate(v), 1e-4F));
}

TEST_CASE("Matrix4x4: inverse round-trips view-projection") {
    const auto view =
        Matrix4x4::look_at_rh({1.5F, 2.0F, 2.8F, 0}, {0, 0.5F, 0, 0}, {0, 1, 0, 0});
    const auto proj = Matrix4x4::perspective_vulkan(1.0472F, 16.0F / 9.0F, 0.1F);
    const auto ortho = Matrix4x4::ortho_vulkan(-10, 10, -10, 10, 0.1F, 100.0F);
    for (const auto& m : {proj * view, ortho * view}) {
        const auto ident = m * m.inverse();
        CHECK(ident.approx_equal(Matrix4x4::identity(), 1e-4F));
        // NDC -> world -> NDC round-trip at a finite depth slice.
        const auto world = m.inverse().transform_point({0.25F, -0.5F, 0.05F, 1.0F});
        const auto ndc = m.transform_point(world);
        CHECK(ndc.approx_equal({0.25F, -0.5F, 0.05F, 1.0F}, 1e-4F));
    }
}
