#pragma once

// vge::render::csm — Cascaded Shadow Map cascade math (Plan §6.1).
// Purely CPU-side helpers: logarithmic split distances and per-cascade
// light view-proj matrices with Gribb-Hartmann frustum planes for the
// meshlet task-shader cull stages. The shadow atlas itself is a plain
// FrameGraph texture (4 quadrant viewports, reverse-Z depth).

#include <cstdint>
#include <vector>

#include <vge/math/matrix4x4.hpp>
#include <vge/math/vector4.hpp>

namespace vulcanGE::render {

inline constexpr std::uint32_t max_shadow_cascades = 4;

struct ShadowCascade {
    math::Matrix4x4 light_vp;         // reverse-Z ortho view-proj
    math::Vector4 frustum_planes[6]{}; // (n, d): inside when n.p + d >= 0
    float split_distance = 0.0F;      // view-space far edge of this cascade
};

// Pure-logarithmic splits (§6.1): split[i] = near * (far/near)^((i+1)/count).
[[nodiscard]] std::vector<float> cascade_splits_log(float z_near, float z_far,
                                                    std::uint32_t count);

// Gribb-Hartmann plane extraction from a column-major view-proj matrix.
// Vulkan clip volume: z in [0, w].
void extract_frustum_planes(const math::Matrix4x4& vp, math::Vector4 (&planes)[6]) noexcept;

// Per-cascade light matrices for a directional light.
//   inv_view_proj: inverse of the camera view-proj (infinite reverse-Z ok —
//   corners are evaluated at finite split distances).
//   camera_near:   camera near plane (ndc z = near / view_dist conversion).
//   shadow_far:    view-space distance covered by the last cascade.
//   light_dir:     world-space direction the light travels (normalized).
// Light-space depth ranges are expanded by caster_margin towards the light so
// off-frustum casters (e.g. objects above the visible slice) still occlude.
[[nodiscard]] std::vector<ShadowCascade> compute_shadow_cascades(
    const math::Matrix4x4& inv_view_proj, float camera_near, float shadow_far,
    const math::Vector4& light_dir, std::uint32_t cascade_count,
    float caster_margin = 20.0F);

} // namespace vulcanGE::render
