#include <vge/render/csm.hpp>

#include <algorithm>
#include <cmath>

namespace vulcanGE::render {

std::vector<float> cascade_splits_log(float z_near, float z_far, std::uint32_t count) {
    std::vector<float> splits(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const float f = static_cast<float>(i + 1) / static_cast<float>(count);
        splits[i] = z_near * std::pow(z_far / z_near, f);
    }
    return splits;
}

void extract_frustum_planes(const math::Matrix4x4& vp, math::Vector4 (&planes)[6]) noexcept {
    auto row = [&](std::size_t i, std::size_t k) { return vp.cols[k][i]; };
    auto set = [&](std::size_t p, float a, float b, float c, float d) {
        const float len = std::sqrt(a * a + b * b + c * c);
        planes[p] = {a / len, b / len, c / len, d / len};
    };
    auto combine = [&](std::size_t p, std::size_t i, float sign) {
        set(p, row(3, 0) + sign * row(i, 0), row(3, 1) + sign * row(i, 1),
            row(3, 2) + sign * row(i, 2), row(3, 3) + sign * row(i, 3));
    };
    combine(0, 0, 1.0F);
    combine(1, 0, -1.0F);
    combine(2, 1, 1.0F);
    combine(3, 1, -1.0F);
    set(4, row(2, 0), row(2, 1), row(2, 2), row(2, 3)); // near: row2 (z >= 0)
    combine(5, 2, -1.0F);
}

std::vector<ShadowCascade> compute_shadow_cascades(const math::Matrix4x4& inv_view_proj,
                                                   float camera_near, float shadow_far,
                                                   const math::Vector4& light_dir,
                                                   std::uint32_t cascade_count,
                                                   float caster_margin) {
    const std::vector<float> splits = cascade_splits_log(camera_near, shadow_far, cascade_count);
    std::vector<ShadowCascade> cascades(cascade_count);

    // World-space frustum corner at view distance d (infinite reverse-Z:
    // ndc z = camera_near / d).
    auto corner_at = [&](float ndc_x, float ndc_y, float view_dist) {
        const float ndc_z = camera_near / view_dist;
        return inv_view_proj.transform_point({ndc_x, ndc_y, ndc_z, 1.0F});
    };

    float slice_near = camera_near;
    for (std::uint32_t i = 0; i < cascade_count; ++i) {
        const float slice_far = splits[i];

        // 8 world-space corners of this frustum slice.
        math::Vector4 corners[8];
        int n = 0;
        for (const float d : {slice_near, slice_far}) {
            for (const float y : {-1.0F, 1.0F}) {
                for (const float x : {-1.0F, 1.0F}) {
                    corners[static_cast<std::size_t>(n++)] = corner_at(x, y, d);
                }
            }
        }

        // Light view: RH look-at from far along -light_dir onto the centroid.
        math::Vector4 centroid{0, 0, 0, 0};
        for (const auto& c : corners) {
            centroid = centroid + c;
        }
        centroid = centroid * (1.0F / 8.0F);
        const math::Vector4 up =
            std::abs(light_dir.y) > 0.99F ? math::Vector4{0, 0, 1, 0} : math::Vector4{0, 1, 0, 0};
        const math::Vector4 eye = centroid - light_dir * 100.0F;
        const math::Matrix4x4 light_view = math::Matrix4x4::look_at_rh(eye, centroid, up);

        // Light-space AABB of the slice corners; expand the depth range
        // towards the light so casters outside the camera frustum occlude.
        float min_x = 1e30F, max_x = -1e30F, min_y = 1e30F, max_y = -1e30F;
        float min_z = 1e30F, max_z = -1e30F;
        for (const auto& c : corners) {
            const math::Vector4 ls = light_view.transform({c.x, c.y, c.z, 1.0F});
            min_x = std::min(min_x, ls.x); max_x = std::max(max_x, ls.x);
            min_y = std::min(min_y, ls.y); max_y = std::max(max_y, ls.y);
            min_z = std::min(min_z, ls.z); max_z = std::max(max_z, ls.z);
        }
        // RH view: geometry in front has z < 0; depth = -z. near/far from
        // the light's near plane, margin towards the light (smaller depth).
        const float depth_near = -max_z - caster_margin;
        const float depth_far = -min_z + caster_margin;

        ShadowCascade& c = cascades[i];
        c.light_vp = math::Matrix4x4::ortho_vulkan(min_x, max_x, min_y, max_y,
                                                   std::max(depth_near, 0.1F), depth_far) *
                     light_view;
        extract_frustum_planes(c.light_vp, c.frustum_planes);
        c.split_distance = slice_far;
        slice_near = slice_far;
    }
    return cascades;
}

} // namespace vulcanGE::render
