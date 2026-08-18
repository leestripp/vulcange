// Clustered compute deferred lighting end-to-end (Plan §5.2): ground plane +
// 3 colored point lights -> meshlet prepass -> froxel light assignment
// (16x9x24, compute) -> G-Buffer -> lighting (per-pixel cluster lookup) ->
// present. Readback proves each light only affects its own region and that
// unlit ground falls back to ambient.

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

struct ClusterPush {
    std::uint32_t lights_buf;
    std::uint32_t cluster_buf;
    std::uint32_t light_count;
    std::uint32_t pad;
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
constexpr std::uint32_t cluster_uints = 65;               // count + 64 indices
constexpr std::uint32_t cluster_count = 16 * 9 * 24;      // froxel grid
constexpr std::uint32_t light_floats = 8;                 // pos, radius, color

void set_viewport_scissor(VkCommandBuffer cmd, VkExtent2D e) {
    const VkViewport vp{0, 0, static_cast<float>(e.width), static_cast<float>(e.height), 0, 1};
    const VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

} // namespace

TEST_CASE("Clustered: colored point lights localized by froxel grid" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_cluster_test", .width = 320, .height = 240});
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

    // ---- Scene: ground plane only ----
    const GroundMesh ground = make_ground_plane(15.0F, -1.2F);
    auto mesh_ground = render::MeshletSet::create(*alloc, ground.vertices, ground.indices, "ground");
    REQUIRE(mesh_ground.has_value());

    // ---- Point lights: red left, green right, blue far-center ----
    struct PointLight {
        float pos[3];
        float radius;
        float color[3];
    };
    const std::array lights{
        PointLight{{-3.0F, 0.0F, -2.0F}, 4.0F, {1.0F, 0.08F, 0.05F}},
        PointLight{{3.0F, 0.0F, -2.0F}, 4.0F, {0.08F, 1.0F, 0.08F}},
        PointLight{{0.0F, 0.0F, -6.0F}, 4.0F, {0.10F, 0.20F, 1.0F}},
    };
    std::vector<std::uint32_t> light_data(lights.size() * light_floats);
    for (std::size_t i = 0; i < lights.size(); ++i) {
        std::memcpy(light_data.data() + i * light_floats, &lights[i], 7 * sizeof(float));
    }
    auto lights_buf = gpu::Buffer::create(*alloc, {.size = light_data.size() * sizeof(std::uint32_t),
                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   .host_visible = true,
                                                   .debug_name = "point_lights"});
    REQUIRE(lights_buf.has_value());
    REQUIRE(lights_buf->write(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(light_data.data()),
        light_data.size() * sizeof(std::uint32_t))));

    auto cluster_buf = gpu::Buffer::create(
        *alloc, {.size = VkDeviceSize{cluster_count} * cluster_uints * sizeof(std::uint32_t),
                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 .host_readable = true, // test reads back the assignment census
                 .debug_name = "clusters"});
    REQUIRE(cluster_buf.has_value());

    // ---- Frame constants ----
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
    // No directional light in this test: point lights + ambient only.
    ubo.light_dir[1] = 1.0F; // points away from all surfaces -> ndl = 0
    ubo.camera_pos[2] = 2.8F;
    render::extract_frustum_planes(vp, *reinterpret_cast<math::Vector4(*)[6]>(ubo.frustum_planes));
    ubo.inv_view_proj = vp.inverse();
    REQUIRE(frame_ubo->write(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(&ubo), sizeof(ubo))));

    // ---- Ground texture (solid gray) ----
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

    auto ground_tex = gpu::Image::create(*alloc, {.width = 4, .height = 4,
                                                  .format = VK_FORMAT_R8G8B8A8_UNORM,
                                                  .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                  .debug_name = "ground_gray"});
    REQUIRE(ground_tex.has_value());
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
                .image = ground_tex->handle(),
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            const VkDependencyInfo dep0{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b0};
            vkCmdPipelineBarrier2(cmd, &dep0);
            const VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {4, 4, 1}};
            vkCmdCopyBufferToImage(cmd, staging->handle(), ground_tex->handle(),
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
                .image = ground_tex->handle(),
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
    auto cluster_cs = load_module("cluster_lights.comp.spv");

    const VkPushConstantRange meshlet_push_range{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0, .size = sizeof(MeshletPush)};
    const VkPushConstantRange lighting_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(LightingPush)};
    const VkPushConstantRange present_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4};
    const VkPushConstantRange cluster_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(ClusterPush)};

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

    auto present_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &fullscreen_vs, .fragment = &present_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {present_push_range},
               .color_formats = {sc->format()},
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "present"});
    REQUIRE(present_pipeline.has_value());

    auto cluster_pipeline = gpu::ComputePipeline::create(
        *dev, {.shader = &cluster_cs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {cluster_push_range},
               .debug_name = "cluster_assign"});
    REQUIRE_MESSAGE(cluster_pipeline.has_value(), cluster_pipeline.error().message());

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
        const render::TextureHandle backbuffer = fg.import_texture(
            "backbuffer", sc->image(image_index), sc->image_view(image_index), sc->format(),
            extent, image_presented[image_index] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                 : VK_IMAGE_LAYOUT_UNDEFINED);
        image_presented[image_index] = true;

        // Pass 1: depth prepass.
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
                                 vkCmdPushConstants(ctx.cmd, prepass_pipeline->layout(),
                                                    VK_SHADER_STAGE_TASK_BIT_EXT |
                                                        VK_SHADER_STAGE_MESH_BIT_EXT |
                                                        VK_SHADER_STAGE_FRAGMENT_BIT,
                                                    0, sizeof(push), &push);
                                 draw_mesh_tasks(ctx.cmd,
                                                 (mesh_ground->meshlet_count() + 31) / 32, 1, 1);
                             });

        // Pass 2: froxel cluster assignment (zero counts, then assign).
        fg.add_compute_pass("cluster_assign", {}, {},
                            [&](const render::PassContext& ctx) {
                                vkCmdFillBuffer(ctx.cmd, cluster_buf->handle(), 0, VK_WHOLE_SIZE,
                                                0);
                                const VkBufferMemoryBarrier2 fill_barrier{
                                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .buffer = cluster_buf->handle(),
                                    .offset = 0,
                                    .size = VK_WHOLE_SIZE};
                                const VkDependencyInfo fill_dep{
                                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .bufferMemoryBarrierCount = 1,
                                    .pBufferMemoryBarriers = &fill_barrier};
                                vkCmdPipelineBarrier2(ctx.cmd, &fill_dep);
                                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                  cluster_pipeline->handle());
                                const VkDescriptorSet set = bindless->set();
                                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        cluster_pipeline->layout(), 0, 1, &set, 0,
                                                        nullptr);
                                const ClusterPush push{.lights_buf = 4,
                                                       .cluster_buf = 5,
                                                       .light_count = lights.size(),
                                                       .pad = 0};
                                vkCmdPushConstants(ctx.cmd, cluster_pipeline->layout(),
                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                   &push);
                                vkCmdDispatch(ctx.cmd, (lights.size() + 63) / 64, 1, 1);
                                const VkBufferMemoryBarrier2 write_barrier{
                                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .buffer = cluster_buf->handle(),
                                    .offset = 0,
                                    .size = VK_WHOLE_SIZE};
                                const VkDependencyInfo write_dep{
                                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .bufferMemoryBarrierCount = 1,
                                    .pBufferMemoryBarriers = &write_barrier};
                                vkCmdPipelineBarrier2(ctx.cmd, &write_dep);
                            });

        // Pass 3: G-Buffer.
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
                                 vkCmdPushConstants(ctx.cmd, gbuffer_pipeline->layout(),
                                                    VK_SHADER_STAGE_TASK_BIT_EXT |
                                                        VK_SHADER_STAGE_MESH_BIT_EXT |
                                                        VK_SHADER_STAGE_FRAGMENT_BIT,
                                                    0, sizeof(push), &push);
                                 draw_mesh_tasks(ctx.cmd,
                                                 (mesh_ground->meshlet_count() + 31) / 32, 1, 1);
                             });

        // Pass 4: lighting with cluster lookup.
        const std::array lighting_reads{gb_normal, gb_albedo, gb_material, depth};
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
                                 const std::uint32_t idx = 4;
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
        CHECK(bindless->register_buffer(mesh_ground->meshlet_buffer(),
                                        mesh_ground->meshlet_count() *
                                            sizeof(render::GpuMeshlet)) == 0);
        CHECK(bindless->register_buffer(mesh_ground->vertex_buffer(),
                                        mesh_ground->vertex_count() * render::vertex_floats *
                                            sizeof(float)) == 1);
        CHECK(bindless->register_buffer(mesh_ground->vertex_indices_buffer(), VK_WHOLE_SIZE) == 2);
        CHECK(bindless->register_buffer(mesh_ground->triangle_indices_buffer(), VK_WHOLE_SIZE) ==
              3);
        CHECK(bindless->register_buffer(lights_buf->handle(), VK_WHOLE_SIZE) == 4);
        CHECK(bindless->register_buffer(cluster_buf->handle(), VK_WHOLE_SIZE) == 5);
        CHECK(bindless->register_texture(ground_tex->view(), sampler) == 0);
        CHECK(bindless->register_texture(fg.texture_view(gb_normal), sampler) == 1);
        CHECK(bindless->register_texture(fg.texture_view(gb_albedo), sampler) == 2);
        CHECK(bindless->register_texture(fg.texture_view(gb_material), sampler) == 3);
        CHECK(bindless->register_texture(fg.texture_view(scene_color), sampler) == 4);
        CHECK(bindless->register_texture(fg.texture_view(depth), sampler) == 5);
        lighting_push.normal_idx = 1;
        lighting_push.albedo_idx = 2;
        lighting_push.material_idx = 3;
        lighting_push.depth_idx = 5;
        lighting_push.shadow_idx = invalid_index;
        lighting_push.light_buf = invalid_index;
        lighting_push.cluster_buf = 5;
        lighting_push.lights_buf = 4;
        lighting_push.light_count = lights.size();
        lighting_push.ao_idx = invalid_index; // AO off
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

    // ---- Readback: per-region color dominance ----
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
        return std::array{static_cast<int>(data[i]), static_cast<int>(data[i + 1]),
                          static_cast<int>(data[i + 2])};
    };
    auto project = [&](float x, float y, float z) {
        const math::Vector4 ndc = vp.transform_point({x, y, z, 1.0F});
        return std::pair{static_cast<std::uint32_t>((ndc.x * 0.5F + 0.5F) *
                                                    static_cast<float>(extent.width)),
                         static_cast<std::uint32_t>((ndc.y * 0.5F + 0.5F) *
                                                    static_cast<float>(extent.height))};
    };

    // Ground points directly beneath each light, plus an unlit corner.
    const auto [rx, ry] = project(-3.0F, -1.2F, -2.0F); // red region
    const auto [gx, gy] = project(3.0F, -1.2F, -2.0F);   // green region
    const auto [bx, by] = project(0.0F, -1.2F, -6.0F);   // blue region
    const auto [dx, dy] = project(0.0F, -1.2F, 0.5F);    // near edge: ambient only
    CHECK(rx < extent.width); CHECK(ry < extent.height);
    CHECK(gx < extent.width); CHECK(gy < extent.height);
    CHECK(bx < extent.width); CHECK(by < extent.height);
    CHECK(dx < extent.width); CHECK(dy < extent.height);
    const auto red = pixel(rx, ry);
    const auto green = pixel(gx, gy);
    const auto blue = pixel(bx, by);
    const auto dim = pixel(dx, dy);
    MESSAGE("red px: ", red[0], ",", red[1], ",", red[2], " | green px: ", green[0], ",",
            green[1], ",", green[2], " | blue px: ", blue[0], ",", blue[1], ",", blue[2],
            " | dim px: ", dim[0], ",", dim[1], ",", dim[2]);

    // Each light dominates only its own region.
    CHECK(red[0] > red[1] * 2);
    CHECK(red[0] > red[2] * 2);
    CHECK(green[1] > green[0] * 2);
    CHECK(green[1] > green[2] * 2);
    CHECK(blue[2] > blue[0] * 2);
    CHECK(blue[2] > blue[1]);
    // Regions are actually lit (well above ambient).
    CHECK(red[0] > 60);
    CHECK(green[1] > 60);
    CHECK(blue[2] > 60);
    // The near-edge ground sees no light sphere -> ambient only (180*0.08~14).
    CHECK(dim[0] < 40);
    CHECK(dim[1] < 40);
    CHECK(dim[2] < 40);

    // ---- Teardown ----
    vkDeviceWaitIdle(dev->handle());
    fg.reset();
    vkDestroySampler(dev->handle(), sampler, nullptr);
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
