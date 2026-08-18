#version 460

// Hardcoded triangle — validates the Frame Graph -> pipeline -> present chain
// before any geometry/mesh infrastructure exists.

layout(location = 0) out vec3 v_color;

const vec2 POSITIONS[3] = vec2[](
    vec2( 0.0, -0.6),
    vec2( 0.6,  0.6),
    vec2(-0.6,  0.6)
);
const vec3 COLORS[3] = vec3[](
    vec3(1.0, 0.2, 0.2),
    vec3(0.2, 1.0, 0.2),
    vec3(0.2, 0.2, 1.0)
);

void main() {
    gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
    v_color = COLORS[gl_VertexIndex];
}
