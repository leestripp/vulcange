#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Fullscreen sky pass (Plan §8): renders the sky-view LUT where the depth
// buffer holds the far value (background). Depth test EQUAL 0 (reverse-Z
// clear), no depth write.

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    vec4 light_dir;
    vec4 camera_pos;
    vec4 frustum_planes[6];
    mat4 inv_view_proj;
} ubo;

layout(set = 0, binding = 2) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    uint sky_tex;
    uint transmittance_tex;
    vec3 sun_dir;
    float sun_angular_radius;
    uint full_w;
    uint full_h;
} push;

layout(location = 0) out vec4 out_color;

void main() {
    // View ray direction (world): unproject the pixel through the inverse
    // projection (dist cancels in the normalize).
    const vec2 ndc = vec2(gl_FragCoord.x / float(push.full_w) * 2.0 - 1.0,
                          gl_FragCoord.y / float(push.full_h) * 2.0 - 1.0);
    const vec3 dir_view = normalize(vec3(ndc.x / ubo.proj[0][0],
                                         ndc.y / ubo.proj[1][1], -1.0));
    const vec3 dir_world = normalize(inverse(mat3(ubo.view)) * dir_view);

    // Sky-view LUT parameterization: y = zenith mu in [-0.2, 1], x = azimuth
    // relative to the sun's vertical plane.
    const float mu_v = clamp(dir_world.y, -0.2, 1.0);
    // Signed azimuth between the horizontal view dir and the sun's plane.
    const vec2 sun_h = normalize(push.sun_dir.xz + vec2(1e-6));
    const vec2 view_h = normalize(dir_world.xz + vec2(1e-6));
    float az = acos(clamp(dot(sun_h, view_h), -1.0, 1.0));
    const float y = (mu_v + 0.2) / 1.2;
    const float x = az / 6.2831853;
    vec3 sky = texture(textures[nonuniformEXT(push.sky_tex)], vec2(x, y)).rgb;

    // Sun disc, computed analytically (the sky-view LUT is too coarse to
    // hold it). Extinction along the sun's zenith path reddens the disc at
    // low elevations.
    const float cos_sun = dot(dir_world, push.sun_dir);
    const float disc = smoothstep(cos(push.sun_angular_radius * 1.5),
                                  cos(push.sun_angular_radius * 0.8), cos_sun);
    if (disc > 0.0) {
        const vec3 t_sun =
            texture(textures[nonuniformEXT(push.transmittance_tex)],
                    vec2(0.001, clamp(push.sun_dir.y * 0.5 + 0.5, 0.0, 1.0))).rgb;
        sky += vec3(1.0, 0.9, 0.8) * disc * 4.0 * t_sun * 0.9;
    }
    out_color = vec4(sky, 1.0);
}
