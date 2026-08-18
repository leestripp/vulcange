#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Deferred lighting resolve (Plan §5.2 Pass 4): one directional light +
// ambient, with CSM shadows (§6.1). G-Buffer, depth and the shadow atlas are
// fetched from the bindless table by push-constant index; world position is
// reconstructed from depth via the inverse view-proj.

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec3 light_dir;
    float pad0;
    vec3 camera_pos;
    float pad1;
    vec4 frustum_planes[6];
    mat4 inv_view_proj;
} ubo;

layout(set = 0, binding = 1) readonly buffer DataBuf { uint data[]; } buffers[];
layout(set = 0, binding = 2) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    vec4 splits;     // view-space far edge per cascade (vec4 first: std430 align)
    uint normal_idx;
    uint albedo_idx;
    uint material_idx;
    uint depth_idx;
    uint shadow_idx; // shadow atlas, 0xFFFFFFFF = shadows off
    uint light_buf;  // per-cascade {light_vp, planes} block, 40 uints each
} push;

layout(location = 0) out vec4 out_color;

float buf_float(uint buf, uint i) {
    return uintBitsToFloat(buffers[nonuniformEXT(buf)].data[i]);
}

void main() {
    const ivec2 px = ivec2(gl_FragCoord.xy);
    const vec4 packed_n = texelFetch(textures[nonuniformEXT(push.normal_idx)], px, 0);
    const vec3 n = packed_n.xyz * 2.0 - 1.0;
    const vec4 albedo_rough = texelFetch(textures[nonuniformEXT(push.albedo_idx)], px, 0);

    // Background (cleared G-Buffer): no geometry written.
    if (dot(n, n) < 0.25) {
        out_color = vec4(0.0);
        return;
    }

    const vec3 N = normalize(n);
    const vec3 L = normalize(-ubo.light_dir);
    const float ndl = max(dot(N, L), 0.0);

    float shadow = 1.0;
    if (push.shadow_idx != 0xFFFFFFFFu) {
        // World position from depth (infinite reverse-Z: ndc z = near / d).
        const vec2 screen = vec2(textureSize(textures[nonuniformEXT(push.depth_idx)], 0));
        const float depth = texelFetch(textures[nonuniformEXT(push.depth_idx)], px, 0).r;
        const vec2 ndc_xy = (gl_FragCoord.xy / screen) * 2.0 - 1.0;
        const vec4 wp = ubo.inv_view_proj * vec4(ndc_xy, depth, 1.0);
        const vec3 world = wp.xyz / wp.w;
        const float view_dist = ubo.proj[3][2] / max(depth, 1e-6); // proj[3][2] = z_near

        uint cascade = 3;
        for (uint i = 0; i < 4; ++i) {
            if (view_dist <= push.splits[i]) {
                cascade = i;
                break;
            }
        }

        const uint lb = push.light_buf;
        const uint lbase = cascade * 40;
        mat4 light_vp;
        for (uint i = 0; i < 16; ++i) {
            light_vp[i / 4][i % 4] = buf_float(lb, lbase + i);
        }
        const vec4 lp = light_vp * vec4(world, 1.0);
        const vec3 lndc = lp.xyz / lp.w;
        const vec2 luv = lndc.xy * 0.5 + 0.5;
        const vec2 atlas_uv = vec2(float(cascade % 2), float(cascade / 2)) * 0.5 + luv * 0.5;

        // Slope-scaled bias in light-ndc units, derived from the cascade
        // ortho matrix (width = 2/vp[0][0], ndc-per-meter = vp[2][2]).
        const float texel_world = 2.0 / (light_vp[0][0] * 2048.0);
        const float bias = texel_world * (1.5 + 3.0 * (1.0 - ndl)) * light_vp[2][2];

        // 3x3 PCF. Reverse-Z: the occluder is in front when its stored depth
        // is LARGER than the receiver's light-space depth.
        const vec2 texel = 1.0 / vec2(textureSize(textures[nonuniformEXT(push.shadow_idx)], 0));
        float lit = 0.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                const float occ = texture(textures[nonuniformEXT(push.shadow_idx)],
                                          atlas_uv + vec2(x, y) * texel).r;
                lit += (occ > lndc.z + bias) ? 0.0 : 1.0;
            }
        }
        shadow = lit / 9.0;
    }

    const vec3 ambient = vec3(0.08);
    out_color = vec4(albedo_rough.rgb * (ambient + vec3(ndl * shadow)), 1.0);
}
