// Meshlet pipeline + Hi-Z occlusion end-to-end (Plan §5.3 / §7.2): two UV
// spheres (occluder in front, occluded behind) -> meshopt meshlets -> depth
// prepass -> Hi-Z pyramid build -> task shader cluster culling (frustum +
// backface cone + Hi-Z occlusion) -> G-Buffer -> lighting -> present.
// Per-object task stats read back from the GPU prove the occluded sphere is
// fully culled in the G-Buffer phase.

#include <chrono>
#include <cmath>
#include <cstring>

#include <doctest/doctest.h>

#include <vge/gpu/gpu.hpp>
#include <vge/math/math.hpp>
#include <vge/platform/window.hpp>
#include <vge/render/bindless_table.hpp>
#include <vge/render/frame_graph.hpp>
#include <vge/render/hiz.hpp>
#include <vge/render/meshlet.hpp>

using namespace vulcanGE;
using namespace std::chrono_literals;

namespace {

constexpr std::uint32_t invalid_index = 0xFFFFFFFFu;

[[nodiscard]] bool vulkan_available() {
    return gpu::Instance::create({.enable_validation = false}).has_value();
}

struct SphereMesh {
    std::vector<float> vertices; // 8 floats per vertex
    std::vector<std::uint32_t> indices;
};

// Y-up UV sphere, pre-transformed to world space (uniform scale + translate).
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
            // CCW from outside (theta+ ring+ order): keeps meshopt cone_axis
            // pointing outward for backface cluster culling.
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
};

struct HizBuildPush {
    std::uint32_t src_tex;
    std::uint32_t src_lod;
    std::uint32_t dst_img;
    std::uint32_t dst_w;
    std::uint32_t dst_h;
};

// Gribb-Hartmann plane extraction from a column-major view-proj matrix.
// Planes are (n, d) with inside == n.p + d >= 0; Vulkan z in [0, w].
void extract_frustum(const math::Matrix4x4& vp, float (&planes)[6][4]) {
    auto row = [&](std::size_t i, std::size_t k) { return vp.cols[k][i]; };
    auto set = [&](std::size_t p, float a, float b, float c, float d) {
        const float len = std::sqrt(a * a + b * b + c * c);
        planes[p][0] = a / len; planes[p][1] = b / len;
        planes[p][2] = c / len; planes[p][3] = d / len;
    };
    auto combine = [&](std::size_t p, std::size_t i, float sign) {
        set(p, row(3, 0) + sign * row(i, 0), row(3, 1) + sign * row(i, 1),
            row(3, 2) + sign * row(i, 2), row(3, 3) + sign * row(i, 3));
    };
    combine(0, 0, 1.0F);
    combine(1, 0, -1.0F);
    combine(2, 1, 1.0F);
    combine(3, 1, -1.0F);
    set(4, row(2, 0), row(2, 1), row(2, 2), row(2, 3)); // near: row2 (z>=0)
    combine(5, 2, -1.0F);
}

void set_viewport_scissor(VkCommandBuffer cmd, VkExtent2D e) {
    const VkViewport vp{0, 0, static_cast<float>(e.width), static_cast<float>(e.height), 0, 1};
    const VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

} // namespace

TEST_CASE("Meshlets: Hi-Z occlusion culling through deferred chain" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_meshlet_test", .width = 320, .height = 240});
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

    // ---- Geometry: occluder (front) + occluded (behind) ----
    const SphereMesh occluder = make_uv_sphere(64, 32, 1.0F, 0.0F, 0.0F, 0.8F);
    const SphereMesh occluded = make_uv_sphere(48, 24, 0.7F, 0.0F, 0.0F, -1.8F);
    auto mesh_a = render::MeshletSet::create(*alloc, occluder.vertices, occluder.indices, "front");
    auto mesh_b =
        render::MeshletSet::create(*alloc, occluded.vertices, occluded.indices, "behind");
    REQUIRE(mesh_a.has_value());
    REQUIRE(mesh_b.has_value());
    MESSAGE("occluder: ", mesh_a->meshlet_count(), " meshlets; occluded: ", mesh_b->meshlet_count(),
            " meshlets");
    CHECK(mesh_a->meshlet_count() > 4);
    CHECK(mesh_b->meshlet_count() > 4);

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
    FrameUBO ubo{};
    ubo.view = view;
    ubo.proj = proj;
    ubo.light_dir[0] = -0.4F; ubo.light_dir[1] = -1.0F; ubo.light_dir[2] = -0.6F;
    ubo.camera_pos[2] = 2.8F;
    extract_frustum(proj * view, ubo.frustum_planes);
    ubo.inv_view_proj = (proj * view).inverse();
    REQUIRE(frame_ubo->write(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(&ubo), sizeof(ubo))));

    // ---- Stats buffers: {tested, emitted} per object per phase ----
    constexpr std::size_t stats_count = 4; // prepass A/B, gbuffer A/B
    std::vector<gpu::Buffer> stats_bufs;
    for (std::size_t i = 0; i < stats_count; ++i) {
        auto b = gpu::Buffer::create(*alloc, {.size = 8,
                                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              .host_readable = true,
                                              .debug_name = "cull_stats"});
        REQUIRE(b.has_value());
        stats_bufs.push_back(std::move(*b));
    }
    const std::array<std::uint32_t, 2> zero_stats{0, 0};

    // ---- Checker texture ----
    constexpr std::uint32_t tex_size = 64;
    std::vector<std::byte> pixels(tex_size * tex_size * 4);
    for (std::uint32_t y = 0; y < tex_size; ++y) {
        for (std::uint32_t x = 0; x < tex_size; ++x) {
            const bool dark = ((x / 8) + (y / 8)) % 2 == 0;
            const std::byte c{dark ? std::byte{40} : std::byte{200}};
            const std::size_t i = (y * tex_size + x) * 4;
            pixels[i] = c; pixels[i + 1] = c; pixels[i + 2] = c;
            pixels[i + 3] = std::byte{255};
        }
    }
    auto checker = gpu::Image::create(*alloc, {.width = tex_size,
                                               .height = tex_size,
                                               .format = VK_FORMAT_R8G8B8A8_UNORM,
                                               .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                               .debug_name = "checker"});
    REQUIRE(checker.has_value());
    auto staging = gpu::Buffer::create(*alloc, {.size = pixels.size(),
                                                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                .host_visible = true,
                                                .debug_name = "checker_staging"});
    REQUIRE(staging.has_value());
    REQUIRE(staging->write(pixels));

    // ---- Hi-Z pyramid ----
    auto pyramid = render::HiZPyramid::create(*dev, *alloc, extent.width, extent.height, "hiz");
    REQUIRE_MESSAGE(pyramid.has_value(), pyramid.error().message());
    MESSAGE("hiz: ", pyramid->mip_count(), " mips, base ",
            pyramid->mip_extent(0).width, "x", pyramid->mip_extent(0).height);

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
            .image = checker->handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep0{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b0};
        vkCmdPipelineBarrier2(cmd, &dep0);
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {tex_size, tex_size, 1}};
        vkCmdCopyBufferToImage(cmd, staging->handle(), checker->handle(),
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
            .image = checker->handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo dep1{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b1};
        vkCmdPipelineBarrier2(cmd, &dep1);
    });

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
    auto hiz_cs = load_module("hiz_downsample.comp.spv");

    const VkPushConstantRange meshlet_push_range{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0, .size = sizeof(MeshletPush)};
    const VkPushConstantRange lighting_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 40};
    const VkPushConstantRange present_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4};
    const VkPushConstantRange hiz_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(HizBuildPush)};

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

    auto hiz_pipeline = gpu::ComputePipeline::create(
        *dev, {.shader = &hiz_cs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {hiz_push_range},
               .debug_name = "hiz_build"});
    REQUIRE_MESSAGE(hiz_pipeline.has_value(), hiz_pipeline.error().message());

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

    auto draw_object = [&](const render::PassContext& ctx, const gpu::GraphicsPipeline& pipe,
                           const render::MeshletSet& mesh, std::uint32_t buf_base,
                           std::uint32_t hiz_tex, std::uint32_t stats_buf) {
        MeshletPush push{};
        std::memcpy(push.model, model.cols, sizeof(push.model));
        push.texture_index = 0; // checker
        push.meshlet_count = mesh.meshlet_count();
        push.meshlet_buf = buf_base + 0;
        push.vertex_buf = buf_base + 1;
        push.vidx_buf = buf_base + 2;
        push.tidx_buf = buf_base + 3;
        push.hiz_tex = hiz_tex;
        push.stats_buf = stats_buf;
        vkCmdPushConstants(ctx.cmd, pipe.layout(),
                           VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), &push);
        draw_mesh_tasks(ctx.cmd, (mesh.meshlet_count() + 31) / 32, 1, 1);
    };

    auto bind_and_draw = [&](const render::PassContext& ctx, const gpu::GraphicsPipeline& pipe,
                             std::uint32_t hiz_tex, std::uint32_t stats_base) {
        set_viewport_scissor(ctx.cmd, extent);
        vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.handle());
        const VkDescriptorSet set = bindless->set();
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout(), 0, 1,
                                &set, 0, nullptr);
        draw_object(ctx, pipe, *mesh_a, 0, hiz_tex, stats_base);
        draw_object(ctx, pipe, *mesh_b, 4, hiz_tex, stats_base + 1);
    };

    for (int frame = 0; frame < 3; ++frame) {
        for (auto& b : stats_bufs) {
            REQUIRE(b.write(std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(zero_stats.data()), sizeof(zero_stats))));
        }

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

        // Pass 1: depth prepass, frustum+cone culling only (no Hi-Z yet).
        const render::Attachment depth_clear{
            .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear = {.depthStencil = {.depth = 0.0F}}};
        fg.add_graphics_pass("meshlet_prepass", {}, {}, depth_clear,
                             [&](const render::PassContext& ctx) {
                                 bind_and_draw(ctx, *prepass_pipeline, invalid_index, 8);
                             });

        // Hi-Z build: compute pass reading the fresh depth buffer.
        fg.add_compute_pass("hiz_build", {&depth, 1}, {},
                            [&](const render::PassContext& ctx) {
                                pyramid->record_build(ctx.cmd, *hiz_pipeline, *bindless,
                                                      fg.texture_view(depth), sampler);
                            });

        // Pass 2: G-Buffer with Hi-Z occlusion culling.
        const std::array gbuffer_atts{
            render::Attachment{.texture = gb_normal, .clear = {.color = {{0, 0, 0, 0}}}},
            render::Attachment{.texture = gb_albedo, .clear = {.color = {{0, 0, 0, 0}}}},
            render::Attachment{.texture = gb_material, .clear = {.color = {{0, 0, 0, 0}}}},
        };
        const render::Attachment depth_load{
            .texture = depth, .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
            .store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE};
        fg.add_graphics_pass("meshlet_gbuffer", {}, gbuffer_atts, depth_load,
                             [&](const render::PassContext& ctx) {
                                 const std::uint32_t hiz_tex =
                                     bindless->register_texture(pyramid->full_view(), sampler);
                                 bind_and_draw(ctx, *gbuffer_pipeline, hiz_tex, 10);
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
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         lighting_pipeline->layout(), 0, 1, &set,
                                                         0, nullptr);
                                 // {gbuffer idx x3, depth, shadow(off),
        //  light_buf, splits}
                                 const std::uint32_t idx[10] = {0, 0, 0, 0, 1, 2, 3, 0xFFFFFFFFu,
                                                                0xFFFFFFFFu, 0};
                                 vkCmdPushConstants(ctx.cmd, lighting_pipeline->layout(),
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(idx),
                                                    idx);
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
        reg_mesh(*mesh_a, 0);
        reg_mesh(*mesh_b, 4);
        for (std::size_t i = 0; i < stats_count; ++i) {
            CHECK(bindless->register_buffer(stats_bufs[i].handle(), 8) == 8 + i);
        }
        CHECK(bindless->register_texture(checker->view(), sampler) == 0);
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

    // ---- Culling stats: the occluded sphere must emit nothing in the
    // Hi-Z-culled G-Buffer phase while the occluder still draws. ----
    auto read_stats = [&](std::size_t i) {
        stats_bufs[i].invalidate();
        std::array<std::uint32_t, 2> v{};
        std::memcpy(v.data(), stats_bufs[i].mapped(), sizeof(v));
        return v;
    };
    const auto pA = read_stats(0);
    const auto pB = read_stats(1);
    const auto gA = read_stats(2);
    const auto gB = read_stats(3);
    MESSAGE("prepass  A: tested=", pA[0], " emitted=", pA[1]);
    MESSAGE("prepass  B: tested=", pB[0], " emitted=", pB[1]);
    MESSAGE("gbuffer  A: tested=", gA[0], " emitted=", gA[1]);
    MESSAGE("gbuffer  B: tested=", gB[0], " emitted=", gB[1]);
    CHECK(pA[0] == mesh_a->meshlet_count());
    CHECK(pB[0] == mesh_b->meshlet_count());
    CHECK(pA[1] > 0);            // occluder draws in prepass
    CHECK(pB[1] > 0);            // occluded sphere draws in prepass (no Hi-Z yet)
    CHECK(gA[1] > 0);            // occluder survives Hi-Z
    CHECK(gB[1] == 0);           // occluded sphere fully culled by Hi-Z
    CHECK(gA[1] <= pA[1]);       // occlusion never emits more than prepass

    // ---- Readback ----
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
    const int center = pixel(extent.width / 2, extent.height / 2);
    const int corner = pixel(2, 2);
    MESSAGE("center px sum: ", center, " corner px sum: ", corner);
    CHECK(center > 0);
    CHECK(corner == 0);
    CHECK(center > corner);

    // ---- Teardown ----
    vkDeviceWaitIdle(dev->handle());
    fg.reset();
    vkDestroySampler(dev->handle(), sampler, nullptr);
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
