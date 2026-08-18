#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Deferred decal (Plan §6.5): fullscreen pass over an oriented decal box.
// World position is reconstructed analytically from the depth buffer
// (reverse-Z: dist = z_near/ndc); fragments inside the decal's local box
// sample the decal texture projected on the box's XY plane and alpha-blend
// into the albedo G-Buffer.

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
    mat4 inv_decal; // world -> decal local (box is [-1,1]^3)
    uint depth_tex;
    uint decal_tex;
    uint full_w;
    uint full_h;
} push;

layout(location = 0) out vec4 out_albedo;

void main() {
    const ivec2 px = ivec2(gl_FragCoord.xy);
    // No "discard" here: it lowers to OpDemoteToHelperInvocation which needs
    // a device feature we don't enable. Alpha 0 is an exact blend no-op.
    const float depth = texelFetch(textures[nonuniformEXT(push.depth_tex)], px, 0).r;
    if (depth <= 1e-6) { // background: no surface to project onto
        out_albedo = vec4(0.0);
        return;
    }
    // Analytic view-space reconstruction (reverse-Z infinite far).
    const float dist = ubo.proj[3][2] / depth;
    const vec2 ndc = vec2(gl_FragCoord.x / float(push.full_w) * 2.0 - 1.0,
                          gl_FragCoord.y / float(push.full_h) * 2.0 - 1.0);
    // proj[1][1] is negative (baked Y-flip): plain division is correct.
    const vec3 pos_view =
        vec3(ndc.x * dist / ubo.proj[0][0], ndc.y * dist / ubo.proj[1][1], -dist);
    const vec4 world_h = inverse(ubo.view) * vec4(pos_view, 1.0);
    const vec3 local = (push.inv_decal * vec4(world_h.xyz, 1.0)).xyz;

    if (any(greaterThan(abs(local), vec3(1.0)))) {
        out_albedo = vec4(0.0);
        return;
    }
    const vec2 decal_uv = local.xy * 0.5 + 0.5;
    out_albedo = texture(textures[nonuniformEXT(push.decal_tex)], decal_uv);
    // Fade the decal at the box's Z faces (projection grazing limit).
    const float z_fade = 1.0 - smoothstep(0.8, 1.0, abs(local.z));
    out_albedo.a *= z_fade;
}
