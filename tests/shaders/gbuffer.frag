#version 460
#extension GL_EXT_nonuniform_qualifier : require

// G-Buffer pass (Plan §5.2 Pass 2):
//   RT0 RGB10A2_UNORM — world normal * 0.5 + 0.5
//   RT1 RGBA8_UNORM   — albedo + roughness
//   RT2 RGBA8_UNORM   — metallic + AO (+ spare)
// Motion vectors (RT3, RG16_SFLOAT) arrive with TAA in a later phase.

layout(set = 0, binding = 2) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    mat4 model;
    uint texture_index;
} push;

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec4 out_normal;
layout(location = 1) out vec4 out_albedo_rough;
layout(location = 2) out vec4 out_metallic_ao;

void main() {
    const vec3 albedo = texture(textures[nonuniformEXT(push.texture_index)], v_uv).rgb;
    out_normal = vec4(normalize(v_normal) * 0.5 + 0.5, 1.0);
    out_albedo_rough = vec4(albedo, 0.5);
    out_metallic_ao = vec4(0.0, 1.0, 0.0, 1.0);
}
