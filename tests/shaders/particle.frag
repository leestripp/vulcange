#version 460

// Soft additive particle fragment (Plan §5.7): radial falloff, fades with
// life. Additive blend state: src.a * rgb added to the HDR scene.

layout(location = 0) in float v_life;
layout(location = 0) out vec4 out_color;

void main() {
    // Particles are small at test scale: flat billboard with life fade is
    // enough; radial falloff comes with the texture atlas in the full VFX pass.
    out_color = vec4(1.0, 0.5, 0.1, 0.35) * v_life;
}
