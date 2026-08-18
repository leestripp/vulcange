// Post-processing pipeline end-to-end (Plan §5.2): ground plane + one
// ultra-bright point light -> meshlet prepass -> cluster assignment ->
// G-Buffer -> lighting into an HDR (R16G16B16A16_SFLOAT) scene -> bloom
// bright-pass + separable blur -> ACES composite -> present. Readbacks prove
// the scene carries HDR (>1) values, bloom energy is localized, and the
// tonemapped output shows the blown-out light with bloom bleed.

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

[[nodiscard]] float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        // Subnormal half: value = mant * 2^-24 (exactly representable).
        const float v = static_cast<float>(mant) * 0x1p-24F;
        return sign != 0u ? -v : v;
    }
    std::uint32_t f;
    if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
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

struct BrightPush {
    std::uint32_t scene_tex;
    std::uint32_t dst_img;
    std::uint32_t full_w;
    std::uint32_t full_h;
    float threshold;
    float knee;
    std::uint32_t pad0;
    std::uint32_t pad1;
};

struct BloomBlurPush {
    std::uint32_t src_tex;
    std::uint32_t dst_img;
    std::uint32_t axis;
    std::uint32_t half_w;
    std::uint32_t half_h;
    std::uint32_t pad0;
    std::uint32_t pad1;
    std::uint32_t pad2;
};

struct CompositePush {
    std::uint32_t scene_tex;
    std::uint32_t bloom_tex;
    float exposure;
    float bloom_strength;
};

constexpr std::uint32_t invalid_index = 0xFFFFFFFFu;
constexpr std::uint32_t cluster_uints = 65;
constexpr std::uint32_t cluster_count = 16 * 9 * 24;

void set_viewport_scissor(VkCommandBuffer cmd, VkExtent2D e) {
    const VkViewport vp{0, 0, static_cast<float>(e.width), static_cast<float>(e.height), 0, 1};
    const VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

} // namespace

TEST_CASE("Post: HDR scene, bloom bloom bleed, ACES composite" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_post_test", .width = 320, .height = 240});
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
    const std::uint32_t half_w = extent.width / 2;
    const std::uint32_t half_h = extent.height / 2;

    auto bindless = render::BindlessTable::create(*dev);
    REQUIRE(bindless.has_value());

    const auto draw_mesh_tasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
        vkGetDeviceProcAddr(dev->handle(), "vkCmdDrawMeshTasksEXT"));
    REQUIRE(draw_mesh_tasks != nullptr);

    // ---- Scene: ground plane, one very bright point light ----
    const GroundMesh ground = make_ground_plane(15.0F, -1.2F);
    auto mesh_ground = render::MeshletSet::create(*alloc, ground.vertices, ground.indices, "gnd");
    REQUIRE(mesh_ground.has_value());

    struct PointLight {
        float pos[3];
        float radius;
        float color[3];
    };
    const std::array lights{
        // Radius 3.0: covers the center/neighbour points but not the far one
        // (dist 3.2) — the far point falls back to ambient only.
        PointLight{{0.0F, 0.0F, -2.0F}, 3.0F, {8.0F, 7.0F, 6.0F}}, // HDR intensity
    };
    std::vector<std::uint32_t> light_data(lights.size() * 8);
    for (std::size_t i = 0; i < lights.size(); ++i) {
        std::memcpy(light_data.data() + i * 8, &lights[i], 7 * sizeof(float));
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
                 .debug_name = "clusters"});
    REQUIRE(cluster_buf.has_value());

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
    ubo.light_dir[1] = 1.0F; // directional points away -> no directional term
    ubo.camera_pos[2] = 2.8F;
    render::extract_frustum_planes(vp, *reinterpret_cast<math::Vector4(*)[6]>(ubo.frustum_planes));
    ubo.inv_view_proj = vp.inverse();
    REQUIRE(frame_ubo->write(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(&ubo), sizeof(ubo))));

    // ---- Ground texture ----
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
    auto cluster_cs = load_module("cluster_lights.comp.spv");
    auto bright_cs = load_module("bright_pass.comp.spv");
    auto bloom_blur_cs = load_module("bloom_blur.comp.spv");
    auto composite_fs = load_module("composite.frag.spv");

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
    const VkPushConstantRange bright_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(BrightPush)};
    const VkPushConstantRange bloom_blur_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(BloomBlurPush)};
    const VkPushConstantRange composite_push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(CompositePush)};

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
               .color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT},
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "lighting_hdr"});
    REQUIRE(lighting_pipeline.has_value());

    auto composite_pipeline = gpu::GraphicsPipeline::create(
        *dev, {.vertex = &fullscreen_vs, .fragment = &composite_fs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {composite_push_range},
               .color_formats = {VK_FORMAT_R8G8B8A8_UNORM},
               .cull_mode = VK_CULL_MODE_NONE,
               .debug_name = "composite"});
    REQUIRE(composite_pipeline.has_value());

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

    auto bright_pipeline = gpu::ComputePipeline::create(
        *dev, {.shader = &bright_cs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {bright_push_range},
               .debug_name = "bright_pass"});
    REQUIRE_MESSAGE(bright_pipeline.has_value(), bright_pipeline.error().message());

    auto bloom_blur_pipeline = gpu::ComputePipeline::create(
        *dev, {.shader = &bloom_blur_cs,
               .set_layouts = {bindless->layout()},
               .push_constant_ranges = {bloom_blur_push_range},
               .debug_name = "bloom_blur"});
    REQUIRE_MESSAGE(bloom_blur_pipeline.has_value(), bloom_blur_pipeline.error().message());

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
    render::TextureHandle bloom_final = render::invalid_texture;
    render::TextureHandle final_ldr = render::invalid_texture;

    LightingPush lighting_push{};

    auto draw_ground = [&](const render::PassContext& ctx, const gpu::GraphicsPipeline& pipe) {
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
        vkCmdPushConstants(ctx.cmd, pipe.layout(),
                           VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), &push);
        draw_mesh_tasks(ctx.cmd, (mesh_ground->meshlet_count() + 31) / 32, 1, 1);
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
            "scene_hdr", {.width = extent.width, .height = extent.height,
                          .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                          .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT});
        const render::TextureHandle bloom_a = fg.create_texture(
            "bloom_a", {.width = half_w, .height = half_h,
                        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT});
        const render::TextureHandle bloom_b = fg.create_texture(
            "bloom_b", {.width = half_w, .height = half_h,
                        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});
        bloom_final = bloom_a;
        final_ldr = fg.create_texture(
            "final_ldr", {.width = extent.width, .height = extent.height,
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
                                 draw_ground(ctx, *prepass_pipeline);
                             });

        // Pass 2: cluster assignment.
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
                                 draw_ground(ctx, *gbuffer_pipeline);
                             });

        // Pass 4: lighting into the HDR scene.
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

        // Pass 5: bloom bright-pass (half-res threshold extract).
        fg.add_compute_pass("bright_pass", {&scene_color, 1}, {&bloom_a, 1},
                            [&](const render::PassContext& ctx) {
                                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                  bright_pipeline->handle());
                                const VkDescriptorSet set = bindless->set();
                                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        bright_pipeline->layout(), 0, 1, &set, 0,
                                                        nullptr);
                                const BrightPush push{.scene_tex = 4,
                                                      .dst_img = 0,
                                                      .full_w = extent.width,
                                                      .full_h = extent.height,
                                                      .threshold = 1.0F,
                                                      .knee = 0.5F,
                                                      .pad0 = 0,
                                                      .pad1 = 0};
                                vkCmdPushConstants(ctx.cmd, bright_pipeline->layout(),
                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                   &push);
                                vkCmdDispatch(ctx.cmd, (half_w + 7) / 8, (half_h + 7) / 8, 1);
                            });

        // Pass 6+7: bloom separable blur (H then V).
        fg.add_compute_pass("bloom_blur_h", {&bloom_a, 1}, {&bloom_b, 1},
                            [&](const render::PassContext& ctx) {
                                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                  bloom_blur_pipeline->handle());
                                const VkDescriptorSet set = bindless->set();
                                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        bloom_blur_pipeline->layout(), 0, 1, &set,
                                                        0, nullptr);
                                const BloomBlurPush push{.src_tex = 6,
                                                         .dst_img = 1,
                                                         .axis = 0,
                                                         .half_w = half_w,
                                                         .half_h = half_h,
                                                         .pad0 = 0,
                                                         .pad1 = 0,
                                                         .pad2 = 0};
                                vkCmdPushConstants(ctx.cmd, bloom_blur_pipeline->layout(),
                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                   &push);
                                vkCmdDispatch(ctx.cmd, (half_w + 7) / 8, (half_h + 7) / 8, 1);
                            });
        fg.add_compute_pass("bloom_blur_v", {&bloom_b, 1}, {&bloom_a, 1},
                            [&](const render::PassContext& ctx) {
                                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                  bloom_blur_pipeline->handle());
                                const VkDescriptorSet set = bindless->set();
                                vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        bloom_blur_pipeline->layout(), 0, 1, &set,
                                                        0, nullptr);
                                const BloomBlurPush push{.src_tex = 7,
                                                         .dst_img = 0,
                                                         .axis = 1,
                                                         .half_w = half_w,
                                                         .half_h = half_h,
                                                         .pad0 = 0,
                                                         .pad1 = 0,
                                                         .pad2 = 0};
                                vkCmdPushConstants(ctx.cmd, bloom_blur_pipeline->layout(),
                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                   &push);
                                vkCmdDispatch(ctx.cmd, (half_w + 7) / 8, (half_h + 7) / 8, 1);
                            });

        // Pass 8: ACES composite into LDR.
        const std::array composite_reads{scene_color, bloom_a};
        const render::Attachment final_att{.texture = final_ldr,
                                           .clear = {.color = {{0, 0, 0, 1}}}};
        fg.add_graphics_pass("composite", composite_reads, {&final_att, 1}, std::nullopt,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   composite_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         composite_pipeline->layout(), 0, 1, &set,
                                                         0, nullptr);
                                 const CompositePush push{.scene_tex = 4,
                                                          .bloom_tex = 6,
                                                          .exposure = 1.0F,
                                                          .bloom_strength = 1.0F};
                                 vkCmdPushConstants(ctx.cmd, composite_pipeline->layout(),
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push),
                                                    &push);
                                 vkCmdDraw(ctx.cmd, 3, 1, 0, 0);
                             });

        const render::Attachment back_att{.texture = backbuffer,
                                          .clear = {.color = {{0, 0, 0, 1}}}};
        fg.add_graphics_pass("present_blit", {&final_ldr, 1}, {&back_att, 1}, std::nullopt,
                             [&](const render::PassContext& ctx) {
                                 set_viewport_scissor(ctx.cmd, extent);
                                 vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                   present_pipeline->handle());
                                 const VkDescriptorSet set = bindless->set();
                                 vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                         present_pipeline->layout(), 0, 1, &set, 0,
                                                         nullptr);
                                 const std::uint32_t idx = 8;
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
        CHECK(bindless->register_texture(gray_tex->view(), sampler) == 0);
        CHECK(bindless->register_texture(fg.texture_view(gb_normal), sampler) == 1);
        CHECK(bindless->register_texture(fg.texture_view(gb_albedo), sampler) == 2);
        CHECK(bindless->register_texture(fg.texture_view(gb_material), sampler) == 3);
        CHECK(bindless->register_texture(fg.texture_view(scene_color), sampler) == 4);
        CHECK(bindless->register_texture(fg.texture_view(depth), sampler) == 5);
        CHECK(bindless->register_texture(fg.texture_view(bloom_a), sampler) == 6);
        CHECK(bindless->register_texture(fg.texture_view(bloom_b), sampler) == 7);
        CHECK(bindless->register_texture(fg.texture_view(final_ldr), sampler) == 8);
        CHECK(bindless->register_storage_image(fg.texture_view(bloom_a)) == 0);
        CHECK(bindless->register_storage_image(fg.texture_view(bloom_b)) == 1);
        lighting_push.normal_idx = 1;
        lighting_push.albedo_idx = 2;
        lighting_push.material_idx = 3;
        lighting_push.depth_idx = 5;
        lighting_push.shadow_idx = invalid_index;
        lighting_push.light_buf = invalid_index;
        lighting_push.cluster_buf = 5;
        lighting_push.lights_buf = 4;
        lighting_push.light_count = lights.size();
        lighting_push.ao_idx = invalid_index;
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

    // ---- Readbacks: HDR scene, bloom texture, final LDR ----
    auto read_image = [&](render::TextureHandle h, std::uint32_t w, std::uint32_t hgt,
                          std::uint32_t bpp, const char* name) {
        auto buf = gpu::Buffer::create(
            *alloc, {.size = VkDeviceSize{w} * hgt * bpp,
                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     .host_readable = true,
                     .debug_name = name});
        REQUIRE(buf.has_value());
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
                .image = fg.texture_image(h),
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                       .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
            vkCmdPipelineBarrier2(cmd, &dep);
            const VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {w, hgt, 1}};
            vkCmdCopyImageToBuffer(cmd, fg.texture_image(h), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   buf->handle(), 1, &region);
        });
        buf->invalidate();
        return buf;
    };

    auto project = [&](float x, float y, float z) {
        const math::Vector4 ndc = vp.transform_point({x, y, z, 1.0F});
        return std::pair{static_cast<std::uint32_t>((ndc.x * 0.5F + 0.5F) *
                                                    static_cast<float>(extent.width)),
                         static_cast<std::uint32_t>((ndc.y * 0.5F + 0.5F) *
                                                    static_cast<float>(extent.height))};
    };

    auto scene_buf = read_image(scene_color, extent.width, extent.height, 8, "scene_rb");
    auto bloom_buf = read_image(bloom_final, half_w, half_h, 8, "bloom_rb");
    auto final_buf = read_image(final_ldr, extent.width, extent.height, 4, "final_rb");

    auto scene_at = [&](float x, float y, float z) {
        const auto [px, py] = project(x, y, z);
        const auto* p = reinterpret_cast<const std::uint16_t*>(scene_buf->mapped());
        const std::size_t i = (static_cast<std::size_t>(py) * extent.width + px) * 4;
        return std::array{half_to_float(p[i]), half_to_float(p[i + 1]), half_to_float(p[i + 2])};
    };
    auto bloom_at = [&](float x, float y, float z) {
        const auto [px, py] = project(x, y, z);
        const std::uint32_t hx = std::min(px / 2, half_w - 1);
        const std::uint32_t hy = std::min(py / 2, half_h - 1);
        const auto* p = reinterpret_cast<const std::uint16_t*>(bloom_buf->mapped());
        const std::size_t i = (static_cast<std::size_t>(hy) * half_w + hx) * 4;
        return half_to_float(p[i]);
    };
    auto final_at = [&](float x, float y, float z) {
        const auto [px, py] = project(x, y, z);
        const auto* p = final_buf->mapped();
        const std::size_t i = (static_cast<std::size_t>(py) * extent.width + px) * 4;
        return static_cast<int>(p[i]) + static_cast<int>(p[i + 1]) + static_cast<int>(p[i + 2]);
    };

    const auto scene_center = scene_at(0.0F, -1.2F, -2.0F);
    const float bloom_center = bloom_at(0.0F, -1.2F, -2.0F);
    const float bloom_far = bloom_at(-3.0F, -1.2F, -2.0F);
    const int fin_center = final_at(0.0F, -1.2F, -2.0F);
    const int fin_neighbor = final_at(1.2F, -1.2F, -2.0F);
    const int fin_far = final_at(-3.0F, -1.2F, -2.0F);
    MESSAGE("scene center (HDR): ", scene_center[0], ",", scene_center[1], ",", scene_center[2],
            " | bloom center: ", bloom_center, " bloom far: ", bloom_far,
            " | final: center ", fin_center, " neighbor ", fin_neighbor, " far ", fin_far);

    CHECK(scene_center[0] > 1.0);          // HDR scene carries >1 values
    CHECK(bloom_center > 0.5F);            // bright-pass extracted the light
    CHECK(bloom_far < 0.05F);              // bloom is localized
    CHECK(fin_center >= 240);              // blown out by scene + bloom
    CHECK(fin_neighbor > fin_far + 40);    // light + bloom bleed
    CHECK(fin_far <= 60);                  // tonemapped ambient stays dark

    // ---- Teardown ----
    vkDeviceWaitIdle(dev->handle());
    fg.reset();
    vkDestroySampler(dev->handle(), sampler, nullptr);
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
