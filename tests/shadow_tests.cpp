// CSM shadow mapping end-to-end (Plan §6.1): sphere caster + ground plane
// receiver -> meshlet depth prepass -> 4-cascade shadow atlas (log splits,
// task-shader light-frustum cluster culling, one pass / quadrant viewports)
// -> G-Buffer -> lighting with PCF shadow sampling -> present. Readback
// proves the ground is dark inside the sphere's shadow and lit outside it.

#include <chrono>
#include <cmath>
#include <cstring>

#include <doctest/doctest.h>

#include <vge/gpu/gpu.hpp>
#include <vge/math/math.hpp>
#include <vge/platform/window.hpp>
#include <vge/render/bindless_table.hpp>
#include <vge/render/csm.hpp>
#include <vge/render/frame_graph.hpp>
#include <vge/render/meshlet.hpp>

using namespace vulcanGE;
using namespace std::chrono_literals;

namespace {

[[nodiscard]] bool vulkan_available() {
    return gpu::Instance::create({.enable_validation = false}).has_value();
}

struct SphereMesh {
    std::vector<float> vertices; // 8 floats per vertex
    std::vector<std::uint32_t> indices;
};

// Y-up UV sphere, pre-transformed to world space (uniform scale + translate).
// Winding is CCW from outside so meshopt cone_axis points outward.
[[nodiscard]] SphereMesh make_uv_sphere(std::uint32_t sectors, std::uint32_t rings, float radius,
                                        float cx, float cy, float cz) {
    SphereMesh m;
    constexpr float pi = 3.14159265358979F;
    for (std::uint32_t r = 0; r <= rings; ++r) {
        const float phi = pi * static_cast<float>(r) / static_cast<float>(rings);
        const float y = std::cos(phi);
        const float rr = std::sin(phi);
        for (std::uint32_t s = 0; s <= sectors; ++s) {
            const float theta = 2.0F * pi * static_cast<float>(s) / static_cast<float>(sectors);
            const float x = rr * std::cos(theta);
            const float z = rr * std::sin(theta);
            m.vertices.insert(m.vertices.end(),
                              {cx + radius * x, cy + radius * y, cz + radius * z, x, y, z,
                               static_cast<float>(s) / static_cast<float>(sectors),
                               static_cast<float>(r) / static_cast<float>(rings)});
        }
    }
    for (std::uint32_t r = 0; r < rings; ++r) {
        for (std::uint32_t s = 0; s < sectors; ++s) {
            const std::uint32_t a = r * (sectors + 1) + s;
            const std::uint32_t b = a + sectors + 1;
            m.indices.insert(m.indices.end(), {a, a + 1, b, a + 1, b + 1, b});
        }
    }
    return m;
}

// Horizontal ground plane (y-up normal, CCW from above), 2 triangles.
[[nodiscard]] SphereMesh make_ground_plane(float half_extent, float y) {
    SphereMesh m;
    const float h = half_extent;
    const float verts[4][8] = {
        {-h, y, -h, 0, 1, 0, 0, 0},
        {h, y, -h, 0, 1, 0, 1, 0},
        {h, y, h, 0, 1, 0, 1, 1},
        {-h, y, h, 0, 1, 0, 0, 1},
    };
    for (const auto& v : verts) {
        m.vertices.insert(m.vertices.end(), v, v + 8);
    }
    m.indices = {0, 2, 1, 0, 3, 2}; // +y normal, CCW from above
    return m;
}

struct FrameUBO {
    math::Matrix4x4 view;
    math::Matrix4x4 proj;
    float light_dir[4];
    float camera_pos[4];
    float frustum_planes[6][4];
    math::Matrix4x4 inv_view_proj;
};

struct MeshletPush {
    float model[16];
    std::uint32_t texture_index;
    std::uint32_t meshlet_count;
    std::uint32_t meshlet_buf;
    std::uint32_t vertex_buf;
    std::uint32_t vidx_buf;
    std::uint32_t tidx_buf;
    std::uint32_t hiz_tex;
    std::uint32_t stats_buf;
};

struct ShadowPush {
    float model[16];
    std::uint32_t meshlet_count;
    std::uint32_t meshlet_buf;
    std::uint32_t vertex_buf;
    std::uint32_t vidx_buf;
    std::uint32_t tidx_buf;
    std::uint32_t light_buf;
    std::uint32_t cascade_index;
};

struct LightingPush {
    float splits[4];
    std::uint32_t normal_idx;
    std::uint32_t albedo_idx;
    std::uint32_t material_idx;
    std::uint32_t depth_idx;
    std::uint32_t shadow_idx;
    std::uint32_t light_buf;
};

constexpr std::uint32_t invalid_index = 0xFFFFFFFFu;
constexpr std::uint32_t shadow_atlas_size = 4096;
constexpr std::uint32_t cascade_size = shadow_atlas_size / 2;

void set_viewport_scissor(VkCommandBuffer cmd, VkExtent2D e) {
    const VkViewport vp{0, 0, static_cast<float>(e.width), static_cast<float>(e.height), 0, 1};
    const VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

} // namespace

TEST_CASE("CSM: sphere shadow on ground plane through deferred chain" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_shadow_test", .width = 320, .height = 240});
    REQUIRE(win.has_value());
    auto exts = platform::Window::required_instance_extensions();
    REQUIRE(exts.has_value());
    auto inst = gpu::Instance::create({.required_extensions = *exts});
    REQUIRE(inst.has_value());
    auto surface = win->create_surface(inst->handle());
    REQUIRE(surface.has_value());
    auto dev = gpu::Device::create(*inst, {.surface = *surface});
    REQUIRE_MESSAGE(dev.has_value(), dev.error().message());
    REQUIRE_MESSAGE(dev->info().features.mesh_shader, "test requires VK_EXT_mesh_shader");
    auto alloc = gpu::Allocator::create(*inst, *dev);
    REQUIRE(alloc.has_value());
    auto sc = gpu::Swapchain::create(*dev, *surface, {.extent = {320, 240}});
    REQUIRE(sc.has_value());
    const VkExtent2D extent = sc->extent();
    CHECK(extent.width == 320);

    auto bindless = render::BindlessTable::create(*dev);
    REQUIRE(bindless.has_value());

    const auto draw_mesh_tasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
        vkGetDeviceProcAddr(dev->handle(), "vkCmdDrawMeshTasksEXT"));
    REQUIRE(draw_mesh_tasks != nullptr);

    // ---- Scene: sphere caster above a ground plane receiver ----
    const SphereMesh sphere = make_uv_sphere(48, 24, 1.0F, 0.0F, 0.5F, 0.8F);
    const SphereMesh ground = make_ground_plane(15.0F, -1.2F);
    auto mesh_sphere = render::MeshletSet::create(*alloc, sphere.vertices, sphere.indices, "caster");
    auto mesh_ground = render::MeshletSet::create(*alloc, ground.vertices, ground.indices, "ground");
    REQUIRE(mesh_sphere.has_value());
    REQUIRE(mesh_ground.has_value());
    MESSAGE("sphere: ", mesh_sphere->meshlet_count(), " meshlets; ground: ",
            mesh_ground->meshlet_count(), " meshlets");

    // ---- Camera / light / cascades ----
    const math::Matrix4x4 view =
        math::Matrix4x4::look_at_rh({0, 0, 2.8F, 0}, {0, 0, 0, 0}, {0, 1, 0, 0});
    const math::Matrix4x4 proj = math::Matrix4x4::perspective_vulkan(
        1.0472F, static_cast<float>(extent.width) / static_cast<float>(extent.height), 0.1F);
    const math::Matrix4x4 vp = proj * view;
    const math::Matrix4x4 inv_vp = vp.inverse();

    // Light travels down and towards -z so the sphere's shadow lands on
    // in-frame ground (z < ~0.7 is visible from this camera).
    const math::Vector4 light_dir =
        math::Vector4{0.30F, -1.0F, -0.35F, 0}.normalized3();
    constexpr float camera_near = 0.1F;
    constexpr float shadow_far = 30.0F;
    const auto cascades = render::compute_shadow_cascades(inv_vp, camera_near, shadow_far,
                                                          light_dir, render::max_shadow_cascades);
    REQUIRE(cascades.size() == render::max_shadow_cascades);
    for (std::size_t i = 0; i < cascades.size(); ++i) {
        MESSAGE("cascade ", i, ": split ", cascades[i].split_distance);
    }

    auto frame_ubo = gpu::Buffer::create(*alloc, {.size = sizeof(FrameUBO),
                                                  .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                  .host_visible = true,
                                                  .debug_name = "frame_ubo"});
    REQUIRE(frame_ubo.has_value());
    FrameUBO ubo{};
    ubo.view = view;
    ubo.proj = proj;
    ubo.light_dir[0] = light_dir.x; ubo.light_dir[1] = light_dir.y; ubo.light_dir[2] = light_dir.z;
    ubo.camera_pos[2] = 2.8F;
    render::extract_frustum_planes(vp, *reinterpret_cast<math::Vector4(*)[6]>(ubo.frustum_planes));
    ubo.inv_view_proj = inv_vp;
    REQUIRE(frame_ubo->write(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(&ubo), sizeof(ubo))));

    // ---- Light buffer: per cascade {light_vp (16), planes (24)} uints ----
    constexpr std::size_t light_uints_per_cascade = 40;
    std::vector<std::uint32_t> light_data(cascades.size() * light_uints_per_cascade);
    for (std::size_t i = 0; i < cascades.size(); ++i) {
        std::uint32_t* dst = light_data.data() + i * light_uints_per_cascade;
        const float* m = &cascades[i].light_vp.cols[0].x;
        for (std::size_t k = 0; k < 16; ++k) {
            std::memcpy(dst + k, m + k, sizeof(float));
        }
        for (std::size_t pl = 0; pl < 6; ++pl) {
            std::memcpy(dst + 16 + pl * 4, &cascades[i].frustum_planes[pl], sizeof(float) * 4);
        }
    }
    auto light_buf = gpu::Buffer::create(*alloc, {.size = light_data.size() * sizeof(std::uint32_t),
                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  .host_visible = true,
                                                  .debug_name = "light_cascades"});
    REQUIRE(light_buf.has_value());
    REQUIRE(light_buf->write(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(light_data.data()),
        light_data.size() * sizeof(std::uint32_t))));

    // ---- Textures: checker (sphere) + solid gray (ground) ----
    auto pool = gpu::CommandPool::create(*dev, dev->info().queues.graphics);
    REQUIRE(pool.has_value());
    auto fence = gpu::Fence::create(*dev);
    REQUIRE(fence.has_value());

    auto one_time_submit = [&](auto&& record) {
        pool->reset();
        auto cmd = pool->allocate();
        REQUIRE(cmd.has_value());
        const VkCommandBufferBeginInfo begin_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        REQUIRE(vkBeginCommandBuffer(*cmd, &begin_ci) == VK_SUCCESS);
        record(*cmd);
        REQUIRE(vkEndCommandBuffer(*cmd) == VK_SUCCESS);
        const VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = *cmd};
        const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                   .commandBufferInfoCount = 1,
                                   .pCommandBufferInfos = &cmd_si};
        REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) == VK_SUCCESS);
        REQUIRE(fence->wait(5s));
        fence->reset();
    };

    auto upload_texture = [&](std::uint32_t size, auto&& pixel_fn, const char* name) {
        std::vector<std::byte> pixels(size * size * 4);
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                pixel_fn(x, y, pixels.data() + (y * size + x) * 4);
            }
        }
        auto img = gpu::Image::create(*alloc, {.width = size, .height = size,
                                               .format = VK_FORMAT_R8G8B8A8_UNORM,
                                               .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                               .debug_name = name});
        REQUIRE(img.has_value());
        auto staging = gpu::Buffer::create(*alloc, {.size = pixels.size(),
                                                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    .host_visible = true,
                                                    .debug_name = "tex_staging"});
        REQUIRE(staging.has_value());
        REQUIRE(staging->write(pixels));
        one_time_submit([&](VkCommandBuffer cmd) {
            const VkImageMemoryBarrier2 b0{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = img->handle(),
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            const VkDependencyInfo dep0{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b0};
            vkCmdPipelineBarrier2(cmd, &dep0);
            const VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {size, size, 1}};
            vkCmdCopyBufferToImage(cmd, staging->handle(), img->handle(),
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            const VkImageMemoryBarrier2 b1{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = img->handle(),
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            const VkDependencyInfo dep1{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b1};
            vkCmdPipelineBarrier2(cmd, &dep1);
        });
        return std::move(*img);
    };

    auto checker = upload_texture(64,
                                  [](std::uint32_t x, std::uint32_t y, std::byte* px) {
                                      const bool dark = ((x / 8) + (y / 8)) % 2 == 0;
                                      const std::byte c{dark ? std::byte{40} : std::byte{200}};
                                      px[0] = c; px[1] = c; px[2] = c; px[3] = std::byte{255};
                                  },
                                  "checker");
    auto ground_tex = upload_texture(4,
                                     [](std::uint32_t, std::uint32_t, std::byte* px) {
                                         px[0] = std::byte{180}; px[1] = std::byte{180};
                                         px[2] = std::byte{180}; px[3] = std::byte{255};
                                     },
                                     "ground_gray");

    const VkSamplerCreateInfo sampler_ci{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 16.0F};
    VkSampler sampler = VK_NULL_HANDLE;
    REQUIRE(vkCreateSampler(dev->handle(), &sampler_ci, nullptr, &sampler) == VK_SUCCESS);

    // ---- Shaders & pipelines ----
    auto load_module = [&](const char* name) {
        auto spv = gpu::load_spirv(std::string(VGE_SHADER_DIR) + "/" + name);
        REQUIRE_MESSAGE(spv.has_value(), spv.error().message());
        auto mod = gpu::ShaderModule::create(*dev, *spv);
        REQUIRE(mod.has_value());
        return std::move(*mod);
    };
    auto task_shader = load_module("meshlet.task.spv");
    auto mesh_shader = load_module("meshlet.mesh.spv");
    auto shadow_task = load_module("shadow.task.spv");
    auto shadow_mesh = load_module("shadow.mesh.spv");
    auto gbuffer_fs = load_module("gbuffer.frag.spv");
    auto fullscreen_vs = load_module("fullscreen.vert.spv");
    auto lighting_fs = load_module("lighting.frag.spv");
    auto present_fs = load_module("present.frag.spv");

    const VkPushConstantRange meshlet_push_range{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0, .size = sizeof(MeshletPush)};
    const VkPushConstantRange shadow_push_range{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
        .offset = 0, .size = sizeof(ShadowPush)};
    const VkPushConstantRange lighting_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(LightingPush)};
    const VkPushConstantRange present_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4};

    auto prepass_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.task = &task_shader, .mesh = &mesh_shader,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {meshlet_push_range},
               .depth_format = VK_FORMAT_D32_SFLOAT,
               .depth_test = true, .depth_write = true,
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "meshlet_prepass"});
    REQUIRE_MESSAGE(prepass_pipeline.has_value(), prepass_pipeline.error().message());

    auto shadow_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.task = &shadow_task, .mesh = &shadow_mesh,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {shadow_push_range},
               .depth_format = VK_FORMAT_D32_SFLOAT,
               .depth_test = true, .depth_write = true,
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "csm_casters"});
    REQUIRE_MESSAGE(shadow_pipeline.has_value(), shadow_pipeline.error().message());

    auto gbuffer_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.task = &task_shader, .mesh = &mesh_shader, .fragment = &gbuffer_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {meshlet_push_range},
               .color_formats = {VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                 VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
               .depth_format = VK_FORMAT_D32_SFLOAT,
               .depth_test = true, .depth_write = false,
               .depth_compare = VK_COMPARE_OP_EQUAL,
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "meshlet_gbuffer"});
    REQUIRE_MESSAGE(gbuffer_pipeline.has_value(), gbuffer_pipeline.error().message());

    auto lighting_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &fullscreen_vs, .fragment = &lighting_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {lighting_push_range},
               .color_formats = {VK_FORMAT_R8G8B8A8_UNORM},
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "lighting"});
    REQUIRE(lighting_pipeline.has_value());

    auto present_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &fullscreen_vs, .fragment = &present_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {present_push_range},
               .color_formats = {sc->format()},
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "present"});
    REQUIRE(present_pipeline.has_value());

    // ---- Frame loop ----
    auto acquire_sem = gpu::Semaphore::create(*dev);
    REQUIRE(acquire_sem.has_value());
    std::vector<gpu::Semaphore> render_sems;
    for (std::uint32_t i = 0; i < sc->image_count(); ++i) {
        auto sem = gpu::Semaphore::create(*dev);
        REQUIRE(sem.has_value());
        render_sems.push_back(std::move(*sem));
    }
    std::vector<bool> image_presented(sc->image_count(), false);

    const math::Matrix4x4 model = math::Matrix4x4::identity();
    render::FrameGraph fg;
    render::TextureHandle scene_color = render::invalid_texture;

    LightingPush lighting_push{};
    for (std::size_t i = 0; i < cascades.size(); ++i) {
        lighting_push.splits[i] = cascades[i].split_distance;
    }

    auto draw_object = [&](const render::PassContext& ctx, const gpu::GraphicsPipeline& pipe,
                           const render::MeshletSet& mesh, std::uint32_t buf_base,
                           std::uint32_t texture_index) {
        MeshletPush push{};
        std::memcpy(push.model, model.cols, sizeof(push.model));
        push.texture_index = texture_index;
        push.meshlet_count = mesh.meshlet_count();
        push.meshlet_buf = buf_base + 0;
        push.vertex_buf = buf_base + 1;
        push.vidx_buf = buf_base + 2;
        push.tidx_buf = buf_base + 3;
        push.hiz_tex = invalid_index;
        push.stats_buf = invalid_index;
        vkCmdPushConstants(ctx.cmd, pipe.layout(),
                           VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), &push);
        draw_mesh_tasks(ctx.cmd, (mesh.meshlet_count() + 31) / 32, 1, 1);
    };

    auto draw_caster = [&](const render::PassContext& ctx, const render::MeshletSet& mesh,
                           std::uint32_t buf_base, std::uint32_t cascade) {
        ShadowPush push{};
        std::memcpy(push.model, model.cols, sizeof(push.model));
        push.meshlet_count = mesh.meshlet_count();
        push.meshlet_buf = buf_base + 0;
        push.vertex_buf = buf_base + 1;
        push.vidx_buf = buf_base + 2;
        push.tidx_buf = buf_base + 3;
        push.light_buf = 8;
        push.cascade_index = cascade;
        vkCmdPushConstants(ctx.cmd, shadow_pipeline->layout(),
                           VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0,
                           sizeof(push), &push);
        draw_mesh_tasks(ctx.cmd, (mesh.meshlet_count() + 31) / 32, 1, 1);
    };

    for (int frame = 0; frame < 3; ++frame) {
        std::uint32_t image_index = 0;
        const VkResult ar =
            vkAcquireNextImageKHR(dev->handle(), sc->handle(), UINT64_MAX,
                                  acquire_sem->handle(), VK_NULL_HANDLE, &image_index);
        REQUIRE((ar == VK_SUCCESS || ar == VK_SUBOPTIMAL_KHR));

        fg.reset();
        const render::TextureHandle depth = fg.create_texture(
            "depth", {.width = extent.width, .height = extent.height,
                      .format = VK_FORMAT_D32_SFLOAT});
        const render::TextureHandle shadow_atlas = fg.create_texture(
            "csm_atlas", {.width = shadow_atlas_size, .height = shadow_atlas_size,
                          .format = VK_FORMAT_D32_SFLOAT,
                          .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT});
        const render::TextureHandle gb_normal = fg.create_texture(
            "gb_normal", {.width = extent.width, .height = extent.height,
                          .format = VK_FORMAT_A2B10G10R10_UNORM_PACK32});
        const render::TextureHandle gb_albedo = fg.create_texture(
            "gb_albedo", {.width = extent.width, .height = extent.height,
                          .format = VK_FORMAT_R8G8B8A8_UNORM});
        const render::TextureHandle gb_material = fg.create_texture(
            "gb_material", {.width = extent.width, .height = extent.height,
                            .format = VK_FORMAT_R8G8B8A8_UNORM});
        scene_color = fg.create_texture(
            "scene_color", {.width = extent.width, .height = extent.height,
                            .format = VK_FORMAT_R8G8B8A8_UNORM,
                            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT});
        const render::TextureHandle backbuffer = fg.import_texture(
            "backbuffer", sc->image(image_index), sc->image_view(image_index), sc->format(),
            extent, image_presented[image_index] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                 : VK_IMAGE_LAYOUT_UNDEFINED);
        image_presented[image_index] = true;

        // Pass 1: camera depth prepass.
        const render::Attachment depth_clear{
            .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear = {.depthStencil = {.depth = 0.0F}}};
        fg.add_graphics_pass("meshlet_prepass", {}, {}, depth_clear,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   prepass_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         prepass_pipeline->layout(), 0, 1, &set, 0,
                                                         nullptr);
                                 draw_object(ctx, *prepass_pipeline, *mesh_sphere, 0, 0);
                                 draw_object(ctx, *prepass_pipeline, *mesh_ground, 4, 1);
                             });

        // Pass 2: CSM atlas — 4 cascade quadrant viewports in one pass.
        const render::Attachment atlas_clear{
            .texture = shadow_atlas, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear = {.depthStencil = {.depth = 0.0F}}};
        fg.add_graphics_pass("csm_casters", {}, {}, atlas_clear,
                             [&](const render::PassContext& ctx) {
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   shadow_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         shadow_pipeline->layout(), 0, 1, &set, 0,
                                                         nullptr);
                                 for (std::uint32_t c = 0; c < render::max_shadow_cascades; ++c) {
                                     const std::uint32_t qx = (c % 2) * cascade_size;
                                     const std::uint32_t qy = (c / 2) * cascade_size;
                                     const VkViewport cvp{static_cast<float>(qx),
                                                          static_cast<float>(qy),
                                                          static_cast<float>(cascade_size),
                                                          static_cast<float>(cascade_size), 0, 1};
                                     const VkRect2D csc{{static_cast<std::int32_t>(qx),
                                                         static_cast<std::int32_t>(qy)},
                                                        {cascade_size, cascade_size}};
                                     vkCmdSetViewport(ctx.cmd, 0, 1, &cvp);
                                     vkCmdSetScissor(ctx.cmd, 0, 1, &csc);
                                     draw_caster(ctx, *mesh_sphere, 0, c);
                                     draw_caster(ctx, *mesh_ground, 4, c);
                                 }
                             });

        // Pass 3: G-Buffer (frustum+cone culling only).
        const std::array gbuffer_atts{
            render::Attachment{.texture = gb_normal, .clear = {.color = {{0, 0, 0, 0}}}},
            render::Attachment{.texture = gb_albedo, .clear = {.color = {{0, 0, 0, 0}}}},
            render::Attachment{.texture = gb_material, .clear = {.color = {{0, 0, 0, 0}}}},
        };
        const render::Attachment depth_load{
            .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE};
        fg.add_graphics_pass("meshlet_gbuffer", {}, gbuffer_atts, depth_load,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   gbuffer_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         gbuffer_pipeline->layout(), 0, 1, &set, 0,
                                                         nullptr);
                                 draw_object(ctx, *gbuffer_pipeline, *mesh_sphere, 0, 0);
                                 draw_object(ctx, *gbuffer_pipeline, *mesh_ground, 4, 1);
                             });

        // Pass 4: lighting with CSM shadow sampling.
        const std::array lighting_reads{gb_normal, gb_albedo, gb_material, depth, shadow_atlas};
        const render::Attachment scene_att{.texture = scene_color,
                                           .clear = {.color = {{0, 0, 0, 0}}}};
        fg.add_graphics_pass("lighting", lighting_reads, {&scene_att, 1}, std::nullopt,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   lighting_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         lighting_pipeline->layout(), 0, 1, &set,
                                                         0, nullptr);
                                 vkCmdPushConstants(ctx.cmd, lighting_pipeline->layout(),
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                    sizeof(lighting_push), &lighting_push);
                                 vkCmdDraw(ctx.cmd, 3, 1, 0, 0);
                             });

        const render::Attachment back_att{.texture = backbuffer,
                                          .clear = {.color = {{0, 0, 0, 1}}}};
        fg.add_graphics_pass("present_blit", {&scene_color, 1}, {&back_att, 1}, std::nullopt,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   present_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         present_pipeline->layout(), 0, 1, &set, 0,
                                                         nullptr);
                                 const std::uint32_t idx = 5;
                                 vkCmdPushConstants(ctx.cmd, present_pipeline->layout(),
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(idx),
                                                    &idx);
                                 vkCmdDraw(ctx.cmd, 3, 1, 0, 0);
                             });
        fg.present(backbuffer);

        auto compiled = fg.compile(*alloc);
        REQUIRE_MESSAGE(compiled.has_value(), compiled.error().message());

        // Descriptor registrations (indices are pushed in the draw callbacks).
        bindless->reset_buffers();
        bindless->reset_textures();
        bindless->reset_storage_images();
        auto reg_mesh = [&](const render::MeshletSet& m, std::uint32_t base) {
            CHECK(bindless->register_buffer(m.meshlet_buffer(),
                                            m.meshlet_count() * sizeof(render::GpuMeshlet)) ==
                  base);
            CHECK(bindless->register_buffer(m.vertex_buffer(),
                                            m.vertex_count() * render::vertex_floats *
                                                sizeof(float)) == base + 1);
            CHECK(bindless->register_buffer(m.vertex_indices_buffer(), VK_WHOLE_SIZE) ==
                  base + 2);
            CHECK(bindless->register_buffer(m.triangle_indices_buffer(), VK_WHOLE_SIZE) ==
                  base + 3);
        };
        reg_mesh(*mesh_sphere, 0);
        reg_mesh(*mesh_ground, 4);
        CHECK(bindless->register_buffer(light_buf->handle(), VK_WHOLE_SIZE) == 8);
        CHECK(bindless->register_texture(checker.view(), sampler) == 0);
        CHECK(bindless->register_texture(ground_tex.view(), sampler) == 1);
        CHECK(bindless->register_texture(fg.texture_view(gb_normal), sampler) == 2);
        CHECK(bindless->register_texture(fg.texture_view(gb_albedo), sampler) == 3);
        CHECK(bindless->register_texture(fg.texture_view(gb_material), sampler) == 4);
        CHECK(bindless->register_texture(fg.texture_view(scene_color), sampler) == 5);
        CHECK(bindless->register_texture(fg.texture_view(depth), sampler) == 6);
        CHECK(bindless->register_texture(fg.texture_view(shadow_atlas), sampler) == 7);
        lighting_push.normal_idx = 2;
        lighting_push.albedo_idx = 3;
        lighting_push.material_idx = 4;
        lighting_push.depth_idx = 6;
        lighting_push.shadow_idx = 7;
        lighting_push.light_buf = 8;
        bindless->bind_frame_ubo(frame_ubo->handle(), sizeof(FrameUBO));

        pool->reset();
        auto cmd = pool->allocate();
        REQUIRE(cmd.has_value());
        const VkCommandBufferBeginInfo begin_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        REQUIRE(vkBeginCommandBuffer(*cmd, &begin_ci) == VK_SUCCESS);
        fg.execute(*cmd);
        REQUIRE(vkEndCommandBuffer(*cmd) == VK_SUCCESS);

        const VkSemaphoreSubmitInfo wait_si{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = acquire_sem->handle(),
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
        const VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = *cmd};
        const VkSemaphoreSubmitInfo signal_si{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = render_sems[image_index].handle(),
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
        const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                   .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait_si,
                                   .commandBufferInfoCount = 1, .pCommandBufferInfos = &cmd_si,
                                   .signalSemaphoreInfoCount = 1,
                                   .pSignalSemaphoreInfos = &signal_si};
        REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) == VK_SUCCESS);
        REQUIRE(fence->wait(5s));
        fence->reset();

        const VkSwapchainKHR swapchain = sc->handle();
        const VkSemaphore render_done = render_sems[image_index].handle();
        const VkPresentInfoKHR present_info{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1, .pWaitSemaphores = &render_done,
            .swapchainCount = 1, .pSwapchains = &swapchain, .pImageIndices = &image_index};
        const VkResult pr = vkQueuePresentKHR(dev->graphics_queue(), &present_info);
        CHECK((pr == VK_SUCCESS || pr == VK_SUBOPTIMAL_KHR));
    }

    // ---- Readback: shadowed ground dark, lit ground bright, sphere lit ----
    auto readback = gpu::Buffer::create(
        *alloc, {.size = VkDeviceSize{extent.width} * extent.height * 4,
                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 .host_readable = true,
                 .debug_name = "readback"});
    REQUIRE(readback.has_value());

    const VkImage scene_img = fg.texture_image(scene_color);
    one_time_submit([&](VkCommandBuffer cmd) {
        const VkImageMemoryBarrier2 b{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = scene_img,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                   .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(cmd, &dep);
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {extent.width, extent.height, 1}};
        vkCmdCopyImageToBuffer(cmd, scene_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               readback->handle(), 1, &region);
    });

    readback->invalidate();
    const std::byte* data = readback->mapped();
    REQUIRE(data != nullptr);
    auto pixel = [&](std::uint32_t x, std::uint32_t y) {
        const std::size_t i = (static_cast<std::size_t>(y) * extent.width + x) * 4;
        return static_cast<int>(data[i]) + static_cast<int>(data[i + 1]) +
               static_cast<int>(data[i + 2]);
    };
    auto project = [&](float x, float y, float z) {
        const math::Vector4 ndc = vp.transform_point({x, y, z, 1.0F});
        return std::pair{static_cast<std::uint32_t>((ndc.x * 0.5F + 0.5F) *
                                                    static_cast<float>(extent.width)),
                         static_cast<std::uint32_t>((ndc.y * 0.5F + 0.5F) *
                                                    static_cast<float>(extent.height))};
    };

    // Sphere center (0, 0.5, 0.8); the shadow lands at (0.51, -1.2, 0.21)
    // along the light ray. A far-left ground point stays lit.
    const auto [sx, sy] = project(0.51F, -1.2F, 0.21F);  // inside shadow
    const auto [lx, ly] = project(-2.5F, -1.2F, -1.0F);  // lit ground
    CHECK(sx < extent.width);
    CHECK(sy < extent.height);
    CHECK(lx < extent.width);
    CHECK(ly < extent.height);
    const int shadowed = pixel(sx, sy);
    const int lit = pixel(lx, ly);
    const int sphere_px = pixel(extent.width / 2, extent.height / 2);
    MESSAGE("shadowed ground px: ", shadowed, " @", sx, ",", sy,
            " | lit ground px: ", lit, " @", lx, ",", ly,
            " | sphere px: ", sphere_px);
    CHECK(lit > 100);                       // lit ground is bright
    CHECK(shadowed < lit / 3);              // shadowed ground: ambient only
    CHECK(sphere_px > 0);                   // sphere renders

    // ---- Teardown ----
    vkDeviceWaitIdle(dev->handle());
    fg.reset();
    vkDestroySampler(dev->handle(), sampler, nullptr);
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
