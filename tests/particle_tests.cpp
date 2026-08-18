// GPU particle foundation end-to-end (Plan §5.7): 256 particles in a storage
// buffer -> compute simulation (gravity + respawn at the emitter) -> additive
// vertex-pulled billboards depth-tested against the ground prepass. Readbacks
// prove particles rise from the emitter and the scene glows above it.

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

struct SimPush {
    std::uint32_t particle_buf;
    std::uint32_t particle_count;
    float dt;
    float time;
    float emitter[3];
    std::uint32_t pad0;
};

struct ParticlePush {
    std::uint32_t particle_buf;
    float size;
    std::uint32_t pad0;
    std::uint32_t pad1;
};

constexpr std::uint32_t invalid_index = 0xFFFFFFFFu;
constexpr std::uint32_t particle_count = 256;

void set_viewport_scissor(VkCommandBuffer cmd, VkExtent2D e) {
    const VkViewport vp{0, 0, static_cast<float>(e.width), static_cast<float>(e.height), 0, 1};
    const VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

} // namespace

TEST_CASE("Particles: GPU fountain rises and glows" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_particle_test", .width = 320, .height = 240});
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

    // ---- Scene: ground (depth occluder) + emitter at the surface ----
    const float emitter_y = -1.2F;
    const GroundMesh ground = make_ground_plane(15.0F, emitter_y);
    auto mesh_ground = render::MeshletSet::create(*alloc, ground.vertices, ground.indices, "gnd");
    REQUIRE(mesh_ground.has_value());

    auto particle_buf = gpu::Buffer::create(
        *alloc, {.size = VkDeviceSize{particle_count} * 8 * sizeof(std::uint32_t),
                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 .host_visible = true,
                 .debug_name = "particles"});
    REQUIRE(particle_buf.has_value());
    {
        // life = 0 everywhere -> every particle respawns on the first sim step.
        std::vector<std::uint32_t> zeros(particle_count * 8, 0);
        REQUIRE(particle_buf->write(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(zeros.data()),
            zeros.size() * sizeof(std::uint32_t))));
    }

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
    ubo.light_dir[1] = -1.0F;
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
        px.fill(std::byte{40});
        for (std::size_t i = 3; i < px.size(); i += 4) {
            px[i] = std::byte{255};
        }
        auto staging = gpu::Buffer::create(*alloc, {.size = px.size(),
                                                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    .host_visible = true,
                                                    .debug_name = "tex_staging"});
        REQUIRE(staging.has_value());
        REQUIRE(staging->write(px));
        auto one_time = [&](auto&& record) {
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
            REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) ==
                    VK_SUCCESS);
            REQUIRE(fence->wait(5s));
            fence->reset();
        };
        one_time([&](VkCommandBuffer cmd) {
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
    auto sim_cs = load_module("particle_sim.comp.spv");
    auto particle_vs = load_module("particle.vert.spv");
    auto particle_ps = load_module("particle.frag.spv");

    const VkPushConstantRange meshlet_push_range{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0, .size = sizeof(MeshletPush)};
    const VkPushConstantRange lighting_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 56};
    const VkPushConstantRange present_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4};
    const VkPushConstantRange sim_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(SimPush)};
    const VkPushConstantRange particle_push_range{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(ParticlePush)};

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

    auto particle_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &particle_vs, .fragment = &particle_ps,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {particle_push_range},
               .color_formats = {VK_FORMAT_R8G8B8A8_UNORM},
               .depth_format = VK_FORMAT_D32_SFLOAT,
               .depth_test = true, .depth_write = false,
               .depth_compare = VK_COMPARE_OP_GREATER,
               .cull_mode = VK_CULL_MODE_NONE,
               .blend = gpu::BlendMode::additive,
               .debug_name = "particles"});
    REQUIRE_MESSAGE(particle_pipeline.has_value(), particle_pipeline.error().message());

    auto present_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &fullscreen_vs, .fragment = &present_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {present_push_range},
               .color_formats = {sc->format()},
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "present"});
    REQUIRE(present_pipeline.has_value());

    auto sim_pipeline = gpu::ComputePipeline::create(
        *dev, {.shader = &sim_cs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {sim_push_range},
               .debug_name = "particle_sim"});
    REQUIRE_MESSAGE(sim_pipeline.has_value(), sim_pipeline.error().message());

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

    for (int frame = 0; frame < 12; ++frame) {
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

        // Pass 1: particle simulation (compute).
        fg.add_compute_pass("particle_sim", {}, {},
                            [&](const render::PassContext& ctx) {
                                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                  sim_pipeline->handle());
                                const VkDescriptorSet set = bindless->set();
                                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        sim_pipeline->layout(), 0, 1, &set, 0,
                                                        nullptr);
                                const SimPush push{.particle_buf = 4,
                                                   .particle_count = particle_count,
                                                   .dt = 1.0F / 60.0F,
                                                   .time = static_cast<float>(frame) / 60.0F,
                                                   .emitter = {0.0F, emitter_y, -2.0F},
                                                   .pad0 = 0};
                                vkCmdPushConstants(ctx.cmd, sim_pipeline->layout(),
                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                   &push);
                                vkCmdDispatch(ctx.cmd, (particle_count + 63) / 64, 1, 1);
                                const VkBufferMemoryBarrier2 sim_barrier{
                                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                    .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .buffer = particle_buf->handle(),
                                    .offset = 0,
                                    .size = VK_WHOLE_SIZE};
                                const VkDependencyInfo sim_dep{
                                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .bufferMemoryBarrierCount = 1,
                                    .pBufferMemoryBarriers = &sim_barrier};
                                vkCmdPipelineBarrier2(ctx.cmd, &sim_dep);
                            });

        // Pass 2: depth prepass (ground occludes particles behind it).
        const render::Attachment depth_clear{
            .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear = {.depthStencil = {.depth = 0.0F}}};
        fg.add_graphics_pass("meshlet_prepass", {}, {}, depth_clear,
                             [&](const render::PassContext& ctx) {
                                 draw_ground(ctx, *prepass_pipeline);
                             });

        // Pass 3: G-Buffer (for a dim ground under the fountain).
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

        // Pass 4: lighting (ambient-only ground).
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

        // Pass 5: particles (additive, depth-tested against the prepass).
        const render::Attachment scene_load{
            .texture = scene_color, .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE};
        const render::Attachment depth_load2{
            .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE};
        fg.add_graphics_pass("particles", {}, {&scene_load, 1}, depth_load2,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   particle_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         particle_pipeline->layout(), 0, 1, &set,
                                                         0, nullptr);
                                 const ParticlePush push{
                                     .particle_buf = 4, .size = 0.06F, .pad0 = 0, .pad1 = 0};
                                 vkCmdPushConstants(ctx.cmd, particle_pipeline->layout(),
                                                    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push),
                                                    &push);
                                 vkCmdDraw(ctx.cmd, particle_count * 6, 1, 0, 0);
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
        CHECK(bindless->register_buffer(particle_buf->handle(), VK_WHOLE_SIZE) == 4);
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

    // ---- Readbacks ----
    // 1. Particle buffer: mean height above the emitter, all finite.
    {
        particle_buf->invalidate();
        const std::uint32_t* p =
            reinterpret_cast<const std::uint32_t*>(particle_buf->mapped());
        auto f = [](std::uint32_t u) { float v; std::memcpy(&v, &u, 4); return v; };
        float mean_y = 0.0F;
        bool finite = true;
        for (std::uint32_t i = 0; i < particle_count; ++i) {
            const float x = f(p[i * 8 + 0]);
            const float y = f(p[i * 8 + 1]);
            const float z = f(p[i * 8 + 2]);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
                std::abs(x) > 1e5F || std::abs(y) > 1e5F || std::abs(z) > 1e5F) {
                finite = false;
            }
            mean_y += y;
        }
        mean_y /= static_cast<float>(particle_count);
        MESSAGE("particles: mean_y=", mean_y, " (emitter_y=", emitter_y, ")");
        CHECK(finite);
        CHECK(mean_y > emitter_y + 0.1F);
    }

    // 2. Scene: glow above the emitter vs dark sky to the side.
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
    auto pixel = [&](float x, float y, float z) {
        const math::Vector4 ndc = vp.transform_point({x, y, z, 1.0F});
        const std::uint32_t px =
            static_cast<std::uint32_t>((ndc.x * 0.5F + 0.5F) * static_cast<float>(extent.width));
        const std::uint32_t py =
            static_cast<std::uint32_t>((ndc.y * 0.5F + 0.5F) * static_cast<float>(extent.height));
        const std::size_t i = (static_cast<std::size_t>(py) * extent.width + px) * 4;
        return static_cast<int>(data[i]) + static_cast<int>(data[i + 1]) +
               static_cast<int>(data[i + 2]);
    };
    const int glow = pixel(0.0F, -0.8F, -2.0F); // in the fountain plume
    const int sky = pixel(-3.0F, -0.2F, -2.0F); // sky away from it
    MESSAGE("glow above emitter: ", glow, " | sky: ", sky);
    CHECK(glow > sky + 10);

    // ---- Teardown ----
    vkDeviceWaitIdle(dev->handle());
    fg.reset();
    vkDestroySampler(dev->handle(), sampler, nullptr);
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
