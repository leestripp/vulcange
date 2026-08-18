#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Vertex-pulled particle billboards (Plan §5.7): no vertex buffers; 6
// vertices per particle expand the center into a camera-facing quad. Writes
// additively into the HDR scene (depth-tested, no depth write).

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    vec4 light_dir;
    vec4 camera_pos;
    vec4 frustum_planes[6];
    mat4 inv_view_proj;
} ubo;

layout(set = 0, binding = 1) readonly buffer DataBuf { uint data[]; } buffers[];

layout(push_constant) uniform Push {
    uint particle_buf;
    float size;
    uint pad0;
    uint pad1;
} push;

layout(location = 0) out float v_life;

void main() {
    const uint pi = gl_VertexIndex / 6u;
    const uint corner = gl_VertexIndex % 6u;
    const uint pb = nonuniformEXT(push.particle_buf);
    const uint base = pi * 8;
    const vec3 center = vec3(uintBitsToFloat(buffers[pb].data[base + 0]),
                             uintBitsToFloat(buffers[pb].data[base + 1]),
                             uintBitsToFloat(buffers[pb].data[base + 2]));
    v_life = clamp(uintBitsToFloat(buffers[pb].data[base + 3]), 0.0, 1.0);

    const vec2 quad[6] = {vec2(-1, -1), vec2(1, -1), vec2(1, 1),
                          vec2(-1, -1), vec2(1, 1),  vec2(-1, 1)};
    // Camera-facing: expand in view space.
    const vec4 center_view = ubo.view * vec4(center, 1.0);
    const vec3 pos_view = center_view.xyz + vec3(quad[corner] * push.size, 0.0);
    gl_Position = ubo.proj * vec4(pos_view, 1.0);
}
