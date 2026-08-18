// Atmospheric sky end-to-end (Plan §8.1): transmittance LUT (256x64) ->
// sky-view LUT (192x108, single scattering Rayleigh+Mie/HG) -> fullscreen
// sky pass into the background (depth EQUAL 0). Readback at a sunset sun
// elevation: zenith is blue (b > r), horizon toward the sun is warm
// (r > b), the sun disc is blindingly bright.

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

struct GroundMesh {
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
};

[[nodiscard]] GroundMesh make_ground_plane(float half_extent, float y) {
    GroundMesh m;
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
    m.indices = {0, 2, 1, 0, 3, 2};
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
    std::uint32_t lod_buf;
    float lod_threshold;
    float lod_px_factor;
};

struct TransmittancePush {
    std::uint32_t dst_img;
    std::uint32_t pad0;
    std::uint32_t pad1;
    std::uint32_t pad2;
};

struct SkyViewPush { // std430: vec3 aligns to 16 (offsets 0,4,16,28 -> 32B)
    std::uint32_t transmittance_tex;
    std::uint32_t dst_img;
    std::uint32_t pad0;
    std::uint32_t pad1;
    float sun_dir[3];
    float sun_angular_radius;
};

struct SkyPush { // std430: vec3 aligns to 16 (offsets 0,4,16,28,32,36 -> 40B)
    std::uint32_t sky_tex;
    std::uint32_t transmittance_tex;
    std::uint32_t pad0;
    std::uint32_t pad1;
    float sun_dir[3];
    float sun_angular_radius;
    std::uint32_t full_w;
    std::uint32_t full_h;
};

constexpr std::uint32_t invalid_index = 0xFFFFFFFFu;

void set_viewport_scissor(VkCommandBuffer cmd, VkExtent2D e) {
    const VkViewport vp{0, 0, static_cast<float>(e.width), static_cast<float>(e.height), 0, 1};
    const VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

} // namespace

TEST_CASE("Sky: zenith blue, sunset horizon warm, sun disc bright" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_sky_test", .width = 320, .height = 240});
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

    auto bindless = render::BindlessTable::create(*dev);
    REQUIRE(bindless.has_value());

    const auto draw_mesh_tasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
        vkGetDeviceProcAddr(dev->handle(), "vkCmdDrawMeshTasksEXT"));
    REQUIRE(draw_mesh_tasks != nullptr);

    // ---- Scene: ground plane only (the sky fills the background) ----
    const GroundMesh ground = make_ground_plane(15.0F, -1.2F);
    auto mesh_ground = render::MeshletSet::create(*alloc, ground.vertices, ground.indices, "gnd");
    REQUIRE(mesh_ground.has_value());

    // Sun low in the -z sky (sunset ahead of the camera).
    const float sun_elev = 0.10F; // ~5.7 degrees above the horizon
    const float sun_dir[3] = {0.0F, sun_elev, -std::sqrt(1.0F - sun_elev * sun_elev)};

    auto frame_ubo = gpu::Buffer::create(*alloc, {.size = sizeof(FrameUBO),
                                                  .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                  .host_visible = true,
                                                  .debug_name = "frame_ubo"});
    REQUIRE(frame_ubo.has_value());
    const math::Matrix4x4 view =
        math::Matrix4x4::look_at_rh({0, 0, 2.8F, 0}, {0, 0, 0, 0}, {0, 1, 0, 0});
    const math::Matrix4x4 proj = math::Matrix4x4::perspective_vulkan(
        1.0472F, static_cast<float>(extent.width) / static_cast<float>(extent.height), 0.1F);
    const math::Matrix4x4 vp = proj * view;
    FrameUBO ubo{};
    ubo.view = view;
    ubo.proj = proj;
    ubo.light_dir[0] = sun_dir[0];
    ubo.light_dir[1] = sun_dir[1];
    ubo.light_dir[2] = sun_dir[2];
    ubo.camera_pos[2] = 2.8F;
    render::extract_frustum_planes(vp, *reinterpret_cast<math::Vector4(*)[6]>(ubo.frustum_planes));
    ubo.inv_view_proj = vp.inverse();
    REQUIRE(frame_ubo->write(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(&ubo), sizeof(ubo))));

    auto pool = gpu::CommandPool::create(*dev, dev->info().queues.graphics);
    REQUIRE(pool.has_value());
    auto fence = gpu::Fence::create(*dev);
    REQUIRE(fence.has_value());

    auto gray_tex = gpu::Image::create(*alloc, {.width = 4, .height = 4,
                                                .format = VK_FORMAT_R8G8B8A8_UNORM,
                                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                .debug_name = "gray"});
    REQUIRE(gray_tex.has_value());
    {
        std::array<std::byte, 4 * 4 * 4> px;
        px.fill(std::byte{120});
        for (std::size_t i = 3; i < px.size(); i += 4) {
            px[i] = std::byte{255};
        }
        auto staging = gpu::Buffer::create(*alloc, {.size = px.size(),
                                                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    .host_visible = true,
                                                    .debug_name = "tex_staging"});
        REQUIRE(staging.has_value());
        REQUIRE(staging->write(px));
        pool->reset();
        auto cmd = pool->allocate();
        REQUIRE(cmd.has_value());
        const VkCommandBufferBeginInfo begin_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        REQUIRE(vkBeginCommandBuffer(*cmd, &begin_ci) == VK_SUCCESS);
        const VkImageMemoryBarrier2 b0{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = gray_tex->handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep0{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b0};
        vkCmdPipelineBarrier2(*cmd, &dep0);
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {4, 4, 1}};
        vkCmdCopyBufferToImage(*cmd, staging->handle(), gray_tex->handle(),
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
            .image = gray_tex->handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep1{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b1};
        vkCmdPipelineBarrier2(*cmd, &dep1);
        REQUIRE(vkEndCommandBuffer(*cmd) == VK_SUCCESS);
        const VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = *cmd};
        const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                   .commandBufferInfoCount = 1,
                                   .pCommandBufferInfos = &cmd_si};
        REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) == VK_SUCCESS);
        REQUIRE(fence->wait(5s));
        fence->reset();
    }

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
    auto gbuffer_fs = load_module("gbuffer.frag.spv");
    auto fullscreen_vs = load_module("fullscreen.vert.spv");
    auto lighting_fs = load_module("lighting.frag.spv");
    auto present_fs = load_module("present.frag.spv");
    auto transmittance_cs = load_module("sky_transmittance.comp.spv");
    auto sky_view_cs = load_module("sky_view.comp.spv");
    auto sky_fs = load_module("sky.frag.spv");

    const VkPushConstantRange meshlet_push_range{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0, .size = sizeof(MeshletPush)};
    const VkPushConstantRange lighting_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 56};
    const VkPushConstantRange present_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4};
    const VkPushConstantRange transmittance_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(TransmittancePush)};
    const VkPushConstantRange sky_view_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(SkyViewPush)};
    const VkPushConstantRange sky_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(SkyPush)};

    auto prepass_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.task = &task_shader, .mesh = &mesh_shader,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {meshlet_push_range},
               .depth_format = VK_FORMAT_D32_SFLOAT,
               .depth_test = true, .depth_write = true,
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "meshlet_prepass"});
    REQUIRE_MESSAGE(prepass_pipeline.has_value(), prepass_pipeline.error().message());

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

    auto sky_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &fullscreen_vs, .fragment = &sky_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {sky_push_range},
               .color_formats = {VK_FORMAT_R8G8B8A8_UNORM},
               .depth_format = VK_FORMAT_D32_SFLOAT,
               .depth_test = true, .depth_write = false,
               .depth_compare = VK_COMPARE_OP_EQUAL, // background == cleared 0.0
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "sky"});
    REQUIRE_MESSAGE(sky_pipeline.has_value(), sky_pipeline.error().message());

    auto present_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &fullscreen_vs, .fragment = &present_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {present_push_range},
               .color_formats = {sc->format()},
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "present"});
    REQUIRE(present_pipeline.has_value());

    auto transmittance_pipeline = gpu::ComputePipeline::create(
        *dev, {.shader = &transmittance_cs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {transmittance_push_range},
               .debug_name = "sky_transmittance"});
    REQUIRE_MESSAGE(transmittance_pipeline.has_value(), transmittance_pipeline.error().message());

    auto sky_view_pipeline = gpu::ComputePipeline::create(
        *dev, {.shader = &sky_view_cs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {sky_view_push_range},
               .debug_name = "sky_view"});
    REQUIRE_MESSAGE(sky_view_pipeline.has_value(), sky_view_pipeline.error().message());

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
    render::TextureHandle sky_view = render::invalid_texture;
    render::TextureHandle transmittance = render::invalid_texture;

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
        transmittance = fg.create_texture(
            "transmittance", {.width = 256, .height = 64,
                              .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                              .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT});
        sky_view = fg.create_texture(
            "sky_view", {.width = 192, .height = 108,
                         .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                         .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT});
        const render::TextureHandle backbuffer = fg.import_texture(
            "backbuffer", sc->image(image_index), sc->image_view(image_index), sc->format(),
            extent, image_presented[image_index] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                 : VK_IMAGE_LAYOUT_UNDEFINED);
        image_presented[image_index] = true;

        auto draw_ground = [&](const render::PassContext& ctx,
                               const gpu::GraphicsPipeline& pipe) {
            set_viewport_scissor(ctx.cmd, extent);
            vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.handle());
            const VkDescriptorSet set = bindless->set();
            vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout(), 0, 1,
                                    &set, 0, nullptr);
            MeshletPush push{};
            std::memcpy(push.model, model.cols, sizeof(push.model));
            push.texture_index = 0;
            push.meshlet_count = mesh_ground->meshlet_count();
            push.meshlet_buf = 0;
            push.vertex_buf = 1;
            push.vidx_buf = 2;
            push.tidx_buf = 3;
            push.hiz_tex = invalid_index;
            push.stats_buf = invalid_index;
            push.lod_buf = invalid_index;
            push.lod_threshold = 0.0F;
            push.lod_px_factor = 0.0F;
            vkCmdPushConstants(ctx.cmd, pipe.layout(),
                               VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(push), &push);
            draw_mesh_tasks(ctx.cmd, (mesh_ground->meshlet_count() + 31) / 32, 1, 1);
        };

        // Pass 1: transmittance LUT.
        fg.add_compute_pass("sky_transmittance", {}, {&transmittance, 1},
                            [&](const render::PassContext& ctx) {
                                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                  transmittance_pipeline->handle());
                                const VkDescriptorSet set = bindless->set();
                                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        transmittance_pipeline->layout(), 0, 1,
                                                        &set, 0, nullptr);
                                const TransmittancePush push{
                                    .dst_img = 0, .pad0 = 0, .pad1 = 0, .pad2 = 0};
                                vkCmdPushConstants(ctx.cmd, transmittance_pipeline->layout(),
                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                   &push);
                                vkCmdDispatch(ctx.cmd, 256 / 8, 64 / 8, 1);
                            });

        // Pass 2: sky-view LUT.
        fg.add_compute_pass("sky_view", {&transmittance, 1}, {&sky_view, 1},
                            [&](const render::PassContext& ctx) {
                                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                  sky_view_pipeline->handle());
                                const VkDescriptorSet set = bindless->set();
                                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        sky_view_pipeline->layout(), 0, 1, &set, 0,
                                                        nullptr);
                                SkyViewPush push{};
                                push.transmittance_tex = 6;
                                push.dst_img = 1;
                                push.sun_dir[0] = sun_dir[0];
                                push.sun_dir[1] = sun_dir[1];
                                push.sun_dir[2] = sun_dir[2];
                                push.sun_angular_radius = 0.01F;
                                vkCmdPushConstants(ctx.cmd, sky_view_pipeline->layout(),
                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                   &push);
                                vkCmdDispatch(ctx.cmd, 192 / 8, (108 + 7) / 8, 1);
                            });

        // Pass 3: depth prepass (ground occludes sky below the horizon).
        const render::Attachment depth_clear{
            .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear = {.depthStencil = {.depth = 0.0F}}};
        fg.add_graphics_pass("meshlet_prepass", {}, {}, depth_clear,
                             [&](const render::PassContext& ctx) {
                                 draw_ground(ctx, *prepass_pipeline);
                             });

        // Pass 4: G-Buffer.
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
                                 draw_ground(ctx, *gbuffer_pipeline);
                             });

        // Pass 5: lighting (ground lit by the low sun).
        const std::array lighting_reads{gb_normal, gb_albedo, gb_material};
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
                                 const std::uint32_t idx[14] = {
                                     0, 0, 0, 0, 1, 2, 3, invalid_index, invalid_index, 0,
                                     invalid_index, invalid_index, 0, invalid_index};
                                 vkCmdPushConstants(ctx.cmd, lighting_pipeline->layout(),
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(idx),
                                                    idx);
                                 vkCmdDraw(ctx.cmd, 3, 1, 0, 0);
                             });

        // Pass 6: sky fills the background (depth EQUAL 0).
        const render::Attachment scene_load{
            .texture = scene_color, .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE};
        const std::array sky_reads{sky_view};
        fg.add_graphics_pass("sky", sky_reads, {&scene_load, 1}, depth_load,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   sky_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         sky_pipeline->layout(), 0, 1, &set, 0,
                                                         nullptr);
                                 SkyPush push{};
                                 push.sky_tex = 7;
                                 push.transmittance_tex = 6;
                                 push.sun_dir[0] = sun_dir[0];
                                 push.sun_dir[1] = sun_dir[1];
                                 push.sun_dir[2] = sun_dir[2];
                                 push.sun_angular_radius = 0.01F;
                                 push.full_w = extent.width;
                                 push.full_h = extent.height;
                                 vkCmdPushConstants(ctx.cmd, sky_pipeline->layout(),
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push),
                                                    &push);
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
                                 const std::uint32_t idx = 4;
                                 vkCmdPushConstants(ctx.cmd, present_pipeline->layout(),
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(idx),
                                                    &idx);
                                 vkCmdDraw(ctx.cmd, 3, 1, 0, 0);
                             });
        fg.present(backbuffer);

        auto compiled = fg.compile(*alloc);
        REQUIRE_MESSAGE(compiled.has_value(), compiled.error().message());

        bindless->reset_buffers();
        bindless->reset_textures();
        bindless->reset_storage_images();
        CHECK(bindless->register_buffer(mesh_ground->meshlet_buffer(),
                                        mesh_ground->meshlet_count() *
                                            sizeof(render::GpuMeshlet)) == 0);
        CHECK(bindless->register_buffer(mesh_ground->vertex_buffer(),
                                        mesh_ground->vertex_count() * render::vertex_floats *
                                            sizeof(float)) == 1);
        CHECK(bindless->register_buffer(mesh_ground->vertex_indices_buffer(), VK_WHOLE_SIZE) == 2);
        CHECK(bindless->register_buffer(mesh_ground->triangle_indices_buffer(), VK_WHOLE_SIZE) ==
              3);
        CHECK(bindless->register_texture(gray_tex->view(), sampler) == 0);
        CHECK(bindless->register_texture(fg.texture_view(gb_normal), sampler) == 1);
        CHECK(bindless->register_texture(fg.texture_view(gb_albedo), sampler) == 2);
        CHECK(bindless->register_texture(fg.texture_view(gb_material), sampler) == 3);
        CHECK(bindless->register_texture(fg.texture_view(scene_color), sampler) == 4);
        CHECK(bindless->register_texture(gray_tex->view(), sampler) == 5); // slot unused here
        CHECK(bindless->register_texture(fg.texture_view(transmittance), sampler) == 6);
        CHECK(bindless->register_texture(fg.texture_view(sky_view), sampler) == 7);
        CHECK(bindless->register_storage_image(fg.texture_view(transmittance)) == 0);
        CHECK(bindless->register_storage_image(fg.texture_view(sky_view)) == 1);
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

    // ---- Readback ----
    auto readback = gpu::Buffer::create(
        *alloc, {.size = VkDeviceSize{extent.width} * extent.height * 4,
                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 .host_readable = true,
                 .debug_name = "readback"});
    REQUIRE(readback.has_value());
    {
        pool->reset();
        auto cmd = pool->allocate();
        REQUIRE(cmd.has_value());
        const VkCommandBufferBeginInfo begin_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        REQUIRE(vkBeginCommandBuffer(*cmd, &begin_ci) == VK_SUCCESS);
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
            .image = fg.texture_image(scene_color),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                   .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(*cmd, &dep);
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {extent.width, extent.height, 1}};
        vkCmdCopyImageToBuffer(*cmd, fg.texture_image(scene_color),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback->handle(), 1,
                               &region);
        REQUIRE(vkEndCommandBuffer(*cmd) == VK_SUCCESS);
        const VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = *cmd};
        const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                   .commandBufferInfoCount = 1,
                                   .pCommandBufferInfos = &cmd_si};
        REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) == VK_SUCCESS);
        REQUIRE(fence->wait(5s));
        fence->reset();
    }
    readback->invalidate();
    const std::byte* data = readback->mapped();
    auto pixel = [&](std::uint32_t px, std::uint32_t py) {
        const std::size_t i = (static_cast<std::size_t>(py) * extent.width + px) * 4;
        return std::array{static_cast<int>(data[i]), static_cast<int>(data[i + 1]),
                          static_cast<int>(data[i + 2])};
    };

    // True horizon (dir.y = 0) sits at py 120; the sun (elevation 0.1) at
    // py ~99. Sample the warm band just above the horizon at py 116 and the
    // disc at py 99. The band py 120-134 below the true horizon is the
    // atmosphere model's opaque ground (no bounce) and stays black there.
    const auto zenith = pixel(160, 10);
    const auto horizon = pixel(160, 116);
    const auto sun = pixel(160, 99);
    MESSAGE("zenith: ", zenith[0], ",", zenith[1], ",", zenith[2],
            " | horizon: ", horizon[0], ",", horizon[1], ",", horizon[2],
            " | sun: ", sun[0], ",", sun[1], ",", sun[2]);

    CHECK(zenith[2] > zenith[0]);   // blue zenith
    CHECK(zenith[2] > 0);           // sky present (not black)
    CHECK(horizon[0] > 100);        // sunset glow band is bright
    CHECK(sun[0] >= 250);           // blinding sun disc
    CHECK(sun[0] > sun[2] + 40);    // ...reddened by extinction

    // Extinction reddens the sun: read the HDR sky-view LUT at the sun
    // texel (mu_v = 0.1 -> y = 0.25*108 = 27, azimuth 0) and verify r > b
    // (the LDR scene clips the disc to pure white, hiding the tint).
    auto lut_readback = gpu::Buffer::create(
        *alloc, {.size = VkDeviceSize{192} * 108 * 8,
                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 .host_readable = true,
                 .debug_name = "lut_rb"});
    REQUIRE(lut_readback.has_value());
    {
        pool->reset();
        auto cmd = pool->allocate();
        REQUIRE(cmd.has_value());
        const VkCommandBufferBeginInfo begin_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        REQUIRE(vkBeginCommandBuffer(*cmd, &begin_ci) == VK_SUCCESS);
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
            .image = fg.texture_image(sky_view),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                   .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(*cmd, &dep);
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {192, 108, 1}};
        vkCmdCopyImageToBuffer(*cmd, fg.texture_image(sky_view),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, lut_readback->handle(), 1,
                               &region);
        REQUIRE(vkEndCommandBuffer(*cmd) == VK_SUCCESS);
        const VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = *cmd};
        const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                   .commandBufferInfoCount = 1,
                                   .pCommandBufferInfos = &cmd_si};
        REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) == VK_SUCCESS);
        REQUIRE(fence->wait(5s));
        fence->reset();
    }
    lut_readback->invalidate();
    auto half_to_float = [](std::uint16_t h) {
        const std::uint32_t sign = (h & 0x8000u) << 16;
        const std::uint32_t e = (h >> 10) & 0x1Fu;
        const std::uint32_t m = h & 0x3FFu;
        std::uint32_t f;
        if (e == 0) {
            const float v = static_cast<float>(m) * 0x1p-24F;
            return sign != 0u ? -v : v;
        }
        f = e == 31 ? (sign | 0x7F800000u | (m << 13)) : (sign | ((e + 112) << 23) | (m << 13));
        float out;
        std::memcpy(&out, &f, 4);
        return out;
    };
    // Debug: read back the transmittance LUT texel at (x~0, mu=0.1 -> y=35).
    auto tr_readback = gpu::Buffer::create(
        *alloc, {.size = VkDeviceSize{256} * 64 * 8,
                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 .host_readable = true,
                 .debug_name = "tr_rb"});
    REQUIRE(tr_readback.has_value());
    {
        pool->reset();
        auto cmd = pool->allocate();
        REQUIRE(cmd.has_value());
        const VkCommandBufferBeginInfo begin_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        REQUIRE(vkBeginCommandBuffer(*cmd, &begin_ci) == VK_SUCCESS);
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
            .image = fg.texture_image(transmittance),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                   .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(*cmd, &dep);
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {256, 64, 1}};
        vkCmdCopyImageToBuffer(*cmd, fg.texture_image(transmittance),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tr_readback->handle(), 1,
                               &region);
        REQUIRE(vkEndCommandBuffer(*cmd) == VK_SUCCESS);
        const VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = *cmd};
        const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                   .commandBufferInfoCount = 1,
                                   .pCommandBufferInfos = &cmd_si};
        REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) == VK_SUCCESS);
        REQUIRE(fence->wait(5s));
        fence->reset();
    }
    tr_readback->invalidate();
    const auto* lut = reinterpret_cast<const std::uint16_t*>(lut_readback->mapped());
    {
        const auto* tr = reinterpret_cast<const std::uint16_t*>(tr_readback->mapped());
        const std::size_t ti = (static_cast<std::size_t>(35) * 256 + 0) * 4; // mu=0.1, ground
        MESSAGE("transmittance LUT (ground, mu=0.1): r=", half_to_float(tr[ti]),
                " g=", half_to_float(tr[ti + 1]), " b=", half_to_float(tr[ti + 2]));
        const std::size_t tj = (static_cast<std::size_t>(57) * 256 + 0) * 4; // mu=0.8
        MESSAGE("transmittance LUT (ground, mu=0.8): r=", half_to_float(tr[tj]),
                " b=", half_to_float(tr[tj + 2]));
    }
    // Warm horizon glow: LUT texel at mu_v ~0.03 (row 20) on the sun's
    // meridian (azimuth 0) — the inscatter there is multiplied by the
    // reddened sun transmittance.
    const std::size_t si = (static_cast<std::size_t>(20) * 192 + 0) * 4;
    const float lut_r = half_to_float(lut[si]);
    const float lut_b = half_to_float(lut[si + 2]);
    MESSAGE("sky LUT horizon glow texel: r=", lut_r, " b=", lut_b);
    CHECK(lut_r > lut_b * 1.4F); // reddened by Rayleigh extinction at sunset
    CHECK(lut_r > 0.05F);        // ...and clearly present

    // ---- Teardown ----
    vkDeviceWaitIdle(dev->handle());
    fg.reset();
    vkDestroySampler(dev->handle(), sampler, nullptr);
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
