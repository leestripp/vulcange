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
    uint shadow_idx;  // shadow atlas, 0xFFFFFFFF = shadows off
    uint light_buf;   // per-cascade {light_vp, planes} block, 40 uints each
    uint cluster_buf; // froxel cluster light lists, 0xFFFFFFFF = clustered off
    uint lights_buf;  // point light array, 8 uints per light
    uint light_count;
    uint ao_idx;      // half-res AO texture, 0xFFFFFFFF = AO off
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

    // World position from depth (infinite reverse-Z: ndc z = near / d).
    // Shared by shadow sampling and clustered lights; evaluated only when at
    // least one of them is enabled (depth_idx is invalid otherwise).
    vec2 screen = vec2(1.0);
    vec3 world = vec3(0.0);
    float view_dist = 0.0;
    const bool need_world = push.shadow_idx != 0xFFFFFFFFu || push.cluster_buf != 0xFFFFFFFFu ||
                            push.ao_idx != 0xFFFFFFFFu;
    if (need_world) {
        screen = vec2(textureSize(textures[nonuniformEXT(push.depth_idx)], 0));
        const float depth = texelFetch(textures[nonuniformEXT(push.depth_idx)], px, 0).r;
        const vec2 ndc_xy = (gl_FragCoord.xy / screen) * 2.0 - 1.0;
        const vec4 wp = ubo.inv_view_proj * vec4(ndc_xy, depth, 1.0);
        world = wp.xyz / wp.w;
        view_dist = ubo.proj[3][2] / max(depth, 1e-6); // proj[3][2] = z_near
    }

    float shadow = 1.0;
    if (push.shadow_idx != 0xFFFFFFFFu) {
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

    // Clustered point lights (Plan §5.2): look up this pixel's froxel and
    // accumulate its light list with smooth distance falloff.
    vec3 point_light = vec3(0.0);
    if (push.cluster_buf != 0xFFFFFFFFu) {
        const ivec3 grid = ivec3(16, 9, 24);
        const float z_near = ubo.proj[3][2];
        const float z_far = 30.0;
        const ivec2 tile = clamp(ivec2(gl_FragCoord.xy / screen * vec2(grid.xy)), ivec2(0),
                                 grid.xy - 1);
        const int slice = clamp(
            int(floor(log(view_dist / z_near) / log(z_far / z_near) * float(grid.z))), 0,
            grid.z - 1);
        const uint ci = (uint(slice) * uint(grid.y) + uint(tile.y)) * uint(grid.x) + uint(tile.x);
        const uint cb = nonuniformEXT(push.cluster_buf);
        const uint base = ci * 65u;
        const uint count = min(buffers[cb].data[base], 64u);
        const uint lights = nonuniformEXT(push.lights_buf);
        for (uint k = 0; k < count; ++k) {
            const uint li = buffers[cb].data[base + 1u + k] * 8u;
            const vec3 lpos = vec3(buf_float(lights, li), buf_float(lights, li + 1),
                                   buf_float(lights, li + 2));
            const float lradius = buf_float(lights, li + 3);
            const vec3 lcol = vec3(buf_float(lights, li + 4), buf_float(lights, li + 5),
                                   buf_float(lights, li + 6));
            const vec3 to_light = lpos - world;
            const float d = length(to_light);
            float att = clamp(1.0 - d / lradius, 0.0, 1.0);
            att *= att;
            const float lndl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
            point_light += lcol * (lndl * att);
        }
    }

    // Ground-truth AO (Plan §6.3): modulates ambient (and baked material AO
    // once materials land).
    float ao = 1.0;
    if (push.ao_idx != 0xFFFFFFFFu) {
        ao = texture(textures[nonuniformEXT(push.ao_idx)], gl_FragCoord.xy / screen).r;
    }

    const vec3 ambient = vec3(0.08) * ao;
    out_color = vec4(albedo_rough.rgb * (ambient + vec3(ndl * shadow) + point_light), 1.0);
}
