#version 460

// Shared mesh vertex stage: vertex pulling from the global geometry SSBO
// (Plan §5.3 — no fixed-function vertex input), per-object transform and
// material index via push constants (§5.3).

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec3 light_dir;
    float pad0;
    vec3 camera_pos;
    float pad1;
} ubo;

layout(set = 0, binding = 1) readonly buffer Vertices { Vertex v[]; } buf;

layout(push_constant) uniform Push {
    mat4 model;
    uint texture_index;
} push;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;

void main() {
    const Vertex vtx = buf.v[gl_VertexIndex];
    const vec4 world = push.model * vec4(vtx.px, vtx.py, vtx.pz, 1.0);
    gl_Position = ubo.proj * ubo.view * world;
    v_normal = mat3(push.model) * vec3(vtx.nx, vtx.ny, vtx.nz);
    v_uv = vec2(vtx.u, vtx.v);
}
