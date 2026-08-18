#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Final composite (Plan §5.2 post pipeline): HDR scene + bloom, ACES-fitted
// tonemap, exposure control. Output is linear; the sRGB swapchain/attachment
// encodes on write.

layout(set = 0, binding = 2) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    uint scene_tex;
    uint bloom_tex;
    float exposure;
    float bloom_strength;
} push;

layout(location = 0) out vec4 out_color;

// ACES fitted (Narkowicz).
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    const ivec2 px = ivec2(gl_FragCoord.xy);
    const vec3 scene = texelFetch(textures[nonuniformEXT(push.scene_tex)], px, 0).rgb;
    const vec3 bloom = texelFetch(textures[nonuniformEXT(push.bloom_tex)],
                                  ivec2(px) / 2, 0).rgb;
    const vec3 hdr = scene + bloom * push.bloom_strength;
    out_color = vec4(aces(hdr * push.exposure), 1.0);
}
