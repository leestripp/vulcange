#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Deferred lighting resolve (Plan §5.2 Pass 4, simplified: one directional
// light + ambient; clustered lights arrive with the compute pass later).
// G-Buffer fetched per-pixel from the bindless table by push-constant index.

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec3 light_dir;
    float pad0;
    vec3 camera_pos;
    float pad1;
} ubo;

layout(set = 0, binding = 2) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    uint normal_idx;
    uint albedo_idx;
    uint material_idx;
} push;

layout(location = 0) out vec4 out_color;

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

    const vec3 L = normalize(-ubo.light_dir);
    const float ndl = max(dot(normalize(n), L), 0.0);
    const vec3 ambient = vec3(0.08);
    out_color = vec4(albedo_rough.rgb * (ambient + vec3(ndl)), 1.0);
}
