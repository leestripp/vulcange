#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Scene color -> swapchain blit (tone mapping / HDR output transform lands
// here with the post stack, Plan §5.2 Pass 7).

layout(set = 0, binding = 2) uniform sampler2D textures[];

layout(push_constant) uniform Push { uint scene_idx; } push;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(textures[nonuniformEXT(push.scene_idx)], v_uv);
}
