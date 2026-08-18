// Meshlet LOD DAG end-to-end (Plan §7.2): sphere -> cluster DAG (meshopt
// partition + locked-border simplification) -> task-shader DAG cut by
// screen-space error. A near camera renders fine level-0 clusters; a far
// camera renders a coarse level with far fewer clusters — while the sphere
// still covers the center pixel at both distances.

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
#include <vge/render/meshlet_lod.hpp>

using namespace vulcanGE;
using namespace std::chrono_literals;

namespace {

[[nodiscard]] bool vulkan_available() {
    return gpu::Instance::create({.enable_validation = false}).has_value();
}

struct SimpleMesh {
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
};

[[nodiscard]] SimpleMesh make_uv_sphere(std::uint32_t sectors, std::uint32_t rings, float radius,
                                        float cx, float cy, float cz) {
    SimpleMesh m;
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

struct LightingPush {
    float splits[4];
    std::uint32_t normal_idx;
    std::uint32_t albedo_idx;
    std::uint32_t material_idx;
    std::uint32_t depth_idx;
    std::uint32_t shadow_idx;
    std::uint32_t light_buf;
    std::uint32_t cluster_buf;
    std::uint32_t lights_buf;
    std::uint32_t light_count;
    std::uint32_t ao_idx;
};

constexpr std::uint32_t invalid_index = 0xFFFFFFFFu;

void set_viewport_scissor(VkCommandBuffer cmd, VkExtent2D e) {
    const VkViewport vp{0, 0, static_cast<float>(e.width), static_cast<float>(e.height), 0, 1};
    const VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

} // namespace

TEST_CASE("MeshletLod: DAG cut renders fine near, coarse far" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_lod_test", .width = 320, .height = 240});
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

    // ---- LOD DAG from a sphere ----
    const SimpleMesh sphere = make_uv_sphere(48, 24, 1.0F, 0.0F, 0.0F, 0.0F);
    auto tree = render::MeshletLodTree::create(*alloc, sphere.vertices, sphere.indices, "sphere");
    REQUIRE_MESSAGE(tree.has_value(), tree.error().message());
    MESSAGE("lod tree: ", tree->cluster_count(), " clusters over ", tree->lod_levels(),
            " levels");
    {
        // Dump per-level error ranges (buffer is persistently mapped).
        const std::uint32_t* li = tree->lod_info_data();
        auto f = [](std::uint32_t u) { float v; std::memcpy(&v, &u, 4); return v; };
        for (std::uint32_t level = 0; level < tree->lod_levels(); ++level) {
            std::uint32_t n = 0;
            float gmin = 1e30F, gmax = 0.0F, pmin = 1e30F, pmax = 0.0F;
            for (std::uint32_t c = 0; c < tree->cluster_count(); ++c) {
                if (li[c * 4 + 2] == level) {
                    ++n;
                    gmin = std::min(gmin, f(li[c * 4 + 0]));
                    gmax = std::max(gmax, f(li[c * 4 + 0]));
                    const float pe = f(li[c * 4 + 1]);
                    if (pe < 1e29F) {
                        pmin = std::min(pmin, pe);
                        pmax = std::max(pmax, pe);
                    }
                }
            }
            MESSAGE("level ", level, ": n=", n, " group_err=[", gmin, "..", gmax,
                    "] parent_err=[", pmin, "..", pmax, "]");
        }
    }

    // ---- Frame resources ----
    auto frame_ubo = gpu::Buffer::create(*alloc, {.size = sizeof(FrameUBO),
                                                  .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                  .host_visible = true,
                                                  .debug_name = "frame_ubo"});
    REQUIRE(frame_ubo.has_value());
    auto stats_buf = gpu::Buffer::create(*alloc, {.size = 8,
                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  .host_readable = true,
                                                  .debug_name = "lod_stats"});
    REQUIRE(stats_buf.has_value());

    // ---- Solid gray texture ----
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

    auto gray_tex = gpu::Image::create(*alloc, {.width = 4, .height = 4,
                                                .format = VK_FORMAT_R8G8B8A8_UNORM,
                                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                .debug_name = "gray"});
    REQUIRE(gray_tex.has_value());
    {
        std::array<std::byte, 4 * 4 * 4> px;
        px.fill(std::byte{180});
        for (std::size_t i = 3; i < px.size(); i += 4) {
            px[i] = std::byte{255};
        }
        auto staging = gpu::Buffer::create(*alloc, {.size = px.size(),
                                                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    .host_visible = true,
                                                    .debug_name = "tex_staging"});
        REQUIRE(staging.has_value());
        REQUIRE(staging->write(px));
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
                .image = gray_tex->handle(),
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            const VkDependencyInfo dep0{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b0};
            vkCmdPipelineBarrier2(cmd, &dep0);
            const VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {4, 4, 1}};
            vkCmdCopyBufferToImage(cmd, staging->handle(), gray_tex->handle(),
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
            vkCmdPipelineBarrier2(cmd, &dep1);
        });
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

    const VkPushConstantRange meshlet_push_range{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0, .size = sizeof(MeshletPush)};
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
               .debug_name = "lod_prepass"});
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
               .debug_name = "lod_gbuffer"});
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

    struct PhaseResult {
        std::uint32_t tested = 0;
        std::uint32_t emitted = 0;
        int center_px = 0;
    };

    const math::Matrix4x4 proj = math::Matrix4x4::perspective_vulkan(
        1.0472F, static_cast<float>(extent.width) / static_cast<float>(extent.height), 0.1F);
    const float lod_px_factor =
        std::abs(proj.cols[0].x) * static_cast<float>(extent.width) * 0.5F;

    auto run_phase = [&](float camera_z) {
        const math::Matrix4x4 view =
            math::Matrix4x4::look_at_rh({0, 0, camera_z, 0}, {0, 0, 0, 0}, {0, 1, 0, 0});
        const math::Matrix4x4 vp = proj * view;
        FrameUBO ubo{};
        ubo.view = view;
        ubo.proj = proj;
        ubo.light_dir[1] = -1.0F;
        ubo.camera_pos[2] = camera_z;
        render::extract_frustum_planes(vp,
                                       *reinterpret_cast<math::Vector4(*)[6]>(ubo.frustum_planes));
        ubo.inv_view_proj = vp.inverse();
        REQUIRE(frame_ubo->write(
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(&ubo), sizeof(ubo))));

        const std::array<std::uint32_t, 2> zero{0, 0};
        REQUIRE(stats_buf->write(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(zero.data()), sizeof(zero))));

        render::TextureHandle scene_color = render::invalid_texture;
        for (int frame = 0; frame < 2; ++frame) {
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
            const render::TextureHandle backbuffer = fg.import_texture(
                "backbuffer", sc->image(image_index), sc->image_view(image_index), sc->format(),
                extent, image_presented[image_index] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                     : VK_IMAGE_LAYOUT_UNDEFINED);
            image_presented[image_index] = true;

            auto draw_tree = [&](const render::PassContext& ctx,
                                 const gpu::GraphicsPipeline& pipe) {
                set_viewport_scissor(ctx.cmd, extent);
                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.handle());
                const VkDescriptorSet set = bindless->set();
                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout(), 0,
                                        1, &set, 0, nullptr);
                MeshletPush push{};
                std::memcpy(push.model, model.cols, sizeof(push.model));
                push.texture_index = 0;
                push.meshlet_count = tree->cluster_count();
                push.meshlet_buf = 0;
                push.vertex_buf = 1;
                push.vidx_buf = 2;
                push.tidx_buf = 3;
                push.hiz_tex = invalid_index;
                push.stats_buf = 5;
                push.lod_buf = 4;
                push.lod_threshold = 1.0F;
                push.lod_px_factor = lod_px_factor;
                vkCmdPushConstants(ctx.cmd, pipe.layout(),
                                   VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                                       VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(push), &push);
                draw_mesh_tasks(ctx.cmd, (tree->cluster_count() + 31) / 32, 1, 1);
            };

            const render::Attachment depth_clear{
                .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                .clear = {.depthStencil = {.depth = 0.0F}}};
            fg.add_graphics_pass("lod_prepass", {}, {}, depth_clear,
                                 [&](const render::PassContext& ctx) {
                                     draw_tree(ctx, *prepass_pipeline);
                                 });

            const std::array gbuffer_atts{
                render::Attachment{.texture = gb_normal, .clear = {.color = {{0, 0, 0, 0}}}},
                render::Attachment{.texture = gb_albedo, .clear = {.color = {{0, 0, 0, 0}}}},
                render::Attachment{.texture = gb_material, .clear = {.color = {{0, 0, 0, 0}}}},
            };
            const render::Attachment depth_load{
                .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE};
            fg.add_graphics_pass("lod_gbuffer", {}, gbuffer_atts, depth_load,
                                 [&](const render::PassContext& ctx) {
                                     draw_tree(ctx, *gbuffer_pipeline);
                                 });

            const std::array lighting_reads{gb_normal, gb_albedo, gb_material};
            const render::Attachment scene_att{.texture = scene_color,
                                               .clear = {.color = {{0, 0, 0, 0}}}};
            fg.add_graphics_pass("lighting", lighting_reads, {&scene_att, 1}, std::nullopt,
                                 [&](const render::PassContext& ctx) {
                                     set_viewport_scissor(ctx.cmd, extent);
                                     vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                       lighting_pipeline->handle());
                                     const VkDescriptorSet set = bindless->set();
                                     vkCmdBindDescriptorSets(ctx.cmd,
                                                             VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                             lighting_pipeline->layout(), 0, 1,
                                                             &set, 0, nullptr);
                                     const std::uint32_t idx[14] = {0, 0, 0, 0, 1, 2, 3,
                                                                    invalid_index, invalid_index, 0,
                                                                    invalid_index, invalid_index, 0,
                                                                    invalid_index};
                                     vkCmdPushConstants(ctx.cmd, lighting_pipeline->layout(),
                                                        VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                        sizeof(idx), idx);
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
                                     vkCmdBindDescriptorSets(ctx.cmd,
                                                             VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                             present_pipeline->layout(), 0, 1,
                                                             &set, 0, nullptr);
                                     const std::uint32_t idx = 4;
                                     vkCmdPushConstants(ctx.cmd, present_pipeline->layout(),
                                                        VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                        sizeof(idx), &idx);
                                     vkCmdDraw(ctx.cmd, 3, 1, 0, 0);
                                 });
            fg.present(backbuffer);

            auto compiled = fg.compile(*alloc);
            REQUIRE_MESSAGE(compiled.has_value(), compiled.error().message());

            bindless->reset_buffers();
            bindless->reset_textures();
            bindless->reset_storage_images();
            CHECK(bindless->register_buffer(tree->meshlet_buffer(),
                                            tree->cluster_count() * sizeof(render::GpuMeshlet)) ==
                  0);
            CHECK(bindless->register_buffer(tree->vertex_buffer(),
                                            tree->vertex_count() * render::vertex_floats *
                                                sizeof(float)) == 1);
            CHECK(bindless->register_buffer(tree->vertex_indices_buffer(), VK_WHOLE_SIZE) == 2);
            CHECK(bindless->register_buffer(tree->triangle_indices_buffer(), VK_WHOLE_SIZE) == 3);
            CHECK(bindless->register_buffer(tree->lod_info_buffer(), VK_WHOLE_SIZE) == 4);
            CHECK(bindless->register_buffer(stats_buf->handle(), 8) == 5);
            CHECK(bindless->register_texture(gray_tex->view(), sampler) == 0);
            CHECK(bindless->register_texture(fg.texture_view(gb_normal), sampler) == 1);
            CHECK(bindless->register_texture(fg.texture_view(gb_albedo), sampler) == 2);
            CHECK(bindless->register_texture(fg.texture_view(gb_material), sampler) == 3);
            CHECK(bindless->register_texture(fg.texture_view(scene_color), sampler) == 4);
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
            REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) ==
                    VK_SUCCESS);
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

        // Stats + center pixel for this camera distance.
        stats_buf->invalidate();
        std::array<std::uint32_t, 2> counts{};
        std::memcpy(counts.data(), stats_buf->mapped(), sizeof(counts));

        auto readback = gpu::Buffer::create(
            *alloc, {.size = VkDeviceSize{extent.width} * extent.height * 4,
                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     .host_readable = true,
                     .debug_name = "readback"});
        REQUIRE(readback.has_value());
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
                .image = fg.texture_image(scene_color),
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                       .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
            vkCmdPipelineBarrier2(cmd, &dep);
            const VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {extent.width, extent.height, 1}};
            vkCmdCopyImageToBuffer(cmd, fg.texture_image(scene_color),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback->handle(), 1,
                                   &region);
        });
        readback->invalidate();
        const std::byte* data = readback->mapped();
        const std::size_t i = (static_cast<std::size_t>(extent.height / 2) * extent.width +
                               extent.width / 2) * 4;
        std::size_t nonzero = 0;
        for (std::size_t px = 0; px < static_cast<std::size_t>(extent.width) * extent.height * 4;
             px += 4) {
            if (data[px] != std::byte{0} || data[px + 1] != std::byte{0} ||
                data[px + 2] != std::byte{0}) {
                ++nonzero;
            }
        }
        MESSAGE("  phase camera_z=", camera_z, ": nonzero pixels ", nonzero, "/",
                extent.width * extent.height);
        return PhaseResult{.tested = counts[0],
                           .emitted = counts[1],
                           .center_px = static_cast<int>(data[i]) + static_cast<int>(data[i + 1]) +
                                        static_cast<int>(data[i + 2])};
    };

    const PhaseResult near = run_phase(2.2F);
    const PhaseResult far = run_phase(14.0F);
    MESSAGE("near (d=2.2): tested=", near.tested, " emitted=", near.emitted,
            " center=", near.center_px);
    MESSAGE("far  (d=14):  tested=", far.tested, " emitted=", far.emitted,
            " center=", far.center_px);

    CHECK(tree->cluster_count() > 30);   // DAG has more clusters than level 0 alone
    CHECK(tree->lod_levels() >= 2);
    // stats accumulate over 2 frames x 2 passes (prepass + gbuffer)
    CHECK(near.tested == 4 * tree->cluster_count());
    CHECK(far.tested == 4 * tree->cluster_count());
    CHECK(near.emitted > far.emitted);   // far renders a coarser DAG cut
    CHECK(far.emitted >= 1);             // but something still renders
    CHECK(near.center_px > 0);           // coverage preserved near
    CHECK(far.center_px > 0);            // ...and far

    // ---- Teardown ----
    vkDeviceWaitIdle(dev->handle());
    fg.reset();
    vkDestroySampler(dev->handle(), sampler, nullptr);
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
