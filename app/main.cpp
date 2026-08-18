#include <print>

#include <vge/math/math.hpp>

int main() {
    using namespace vulcanGE::math;

    std::println("vulcanGE sandbox — Phase 1 bootstrap");
    std::println("  alignof(Vector4)   = {}", alignof(Vector4));
    std::println("  alignof(Matrix4x4) = {}", alignof(Matrix4x4));

    const auto proj = Matrix4x4::perspective_vulkan(1.0472F, 16.0F / 9.0F, 0.1F);
    const auto ndc = proj.transform_point({0.0F, 0.0F, -0.1F, 0.0F});
    std::println("  reverse-Z check: near plane -> depth {:.4f} (expect 1.0)", ndc.z);

    const auto far_ndc = proj.transform_point({0.0F, 0.0F, -100000.0F, 0.0F});
    std::println("  reverse-Z check: 100 km     -> depth {:.6f} (expect ~0.0)", far_ndc.z);

    return 0;
}
