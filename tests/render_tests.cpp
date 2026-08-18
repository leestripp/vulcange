// Frame Graph end-to-end test: clear + triangle -> swapchain -> present.
// Runs on the dev machine's real driver (RX 7800 XT / RADV); skips gracefully
// when no Vulkan driver is present.

#include <chrono>

#include <doctest/doctest.h>

#include <vge/gpu/gpu.hpp>
#include <vge/platform/window.hpp>
#include <vge/render/frame_graph.hpp>

using namespace vulcanGE;
using namespace std::chrono_literals;

namespace {

[[nodiscard]] bool vulkan_available() {
    return gpu::Instance::create({.enable_validation = false}).has_value();
}

} // namespace

TEST_CASE("FrameGraph: clear + triangle to swapchain, 3 frames" *
          doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_fg_test", .width = 320, .height = 240});
    REQUIRE(win.has_value());
    auto exts = platform::Window::required_instance_extensions();
    REQUIRE(exts.has_value());
    auto inst = gpu::Instance::create({.required_extensions = *exts});
    REQUIRE(inst.has_value());
    auto surface = win->create_surface(inst->handle());
    REQUIRE(surface.has_value());
    auto dev = gpu::Device::create(*inst, {.surface = *surface});
    REQUIRE_MESSAGE(dev.has_value(), dev.error().message());
    auto alloc = gpu::Allocator::create(*inst, *dev);
    REQUIRE_MESSAGE(alloc.has_value(), alloc.error().message());
    auto sc = gpu::Swapchain::create(*dev, *surface, {});
    REQUIRE_MESSAGE(sc.has_value(), sc.error().message());

    // Shaders (compiled at build time by glslangValidator, cmake/VgeShaders.cmake).
    auto vs_spv = gpu::load_spirv(VGE_SHADER_DIR "/triangle.vert.spv");
    REQUIRE_MESSAGE(vs_spv.has_value(), vs_spv.error().message());
    auto fs_spv = gpu::load_spirv(VGE_SHADER_DIR "/triangle.frag.spv");
    REQUIRE_MESSAGE(fs_spv.has_value(), fs_spv.error().message());
    auto vs = gpu::ShaderModule::create(*dev, *vs_spv);
    REQUIRE(vs.has_value());
    auto fs = gpu::ShaderModule::create(*dev, *fs_spv);
    REQUIRE(fs.has_value());
    auto pipe = gpu::GraphicsPipeline::create(*dev, {.vertex = &*vs,
                                                     .fragment = &*fs,
                                                     .color_formats = {sc->format()},
                                                     .cull_mode = VK_CULL_MODE_NONE,
                                                     .debug_name = "triangle"});
    REQUIRE_MESSAGE(pipe.has_value(), pipe.error().message());

    auto pool = gpu::CommandPool::create(*dev, dev->info().queues.graphics);
    REQUIRE(pool.has_value());
    auto acquire_sem = gpu::Semaphore::create(*dev);
    REQUIRE(acquire_sem.has_value());
    // Present-wait semaphores: one per swapchain image. The presentation
    // engine holds the semaphore until the image is re-acquired, so a single
    // shared semaphore would be re-signaled while still in use
    // (VUID-vkQueueSubmit2-semaphore-03868; Khronos swapchain_semaphore_reuse
    // guide). Plan §5.4 "semaphore recycling" encodes this pattern.
    std::vector<gpu::Semaphore> render_sems;
    for (std::uint32_t i = 0; i < sc->image_count(); ++i) {
        auto sem = gpu::Semaphore::create(*dev);
        REQUIRE(sem.has_value());
        render_sems.push_back(std::move(*sem));
    }
    auto fence = gpu::Fence::create(*dev);
    REQUIRE(fence.has_value());

    // Layout is tracked per swapchain image, not per frame: each image is
    // UNDEFINED until its first transition, PRESENT_SRC_KHR afterwards.
    std::vector<bool> image_presented(sc->image_count(), false);

    for (int frame = 0; frame < 3; ++frame) {
        std::uint32_t image_index = 0;
        const VkResult ar =
            vkAcquireNextImageKHR(dev->handle(), sc->handle(), UINT64_MAX,
                                  acquire_sem->handle(), VK_NULL_HANDLE, &image_index);
        REQUIRE((ar == VK_SUCCESS || ar == VK_SUBOPTIMAL_KHR));

        pool->reset();
        auto cmd = pool->allocate();
        REQUIRE(cmd.has_value());
        const VkCommandBufferBeginInfo begin_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        REQUIRE(vkBeginCommandBuffer(*cmd, &begin_ci) == VK_SUCCESS);

        render::FrameGraph fg;
        const render::TextureHandle backbuffer = fg.import_texture(
            "backbuffer", sc->image(image_index), sc->image_view(image_index), sc->format(),
            sc->extent(),
            image_presented[image_index] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                         : VK_IMAGE_LAYOUT_UNDEFINED);
        image_presented[image_index] = true;

        const render::Attachment color{
            .texture = backbuffer,
            .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear = {.color = {{0.05F, 0.05F, 0.10F, 1.0F}}},
        };
        fg.add_graphics_pass(
            "triangle", {}, {&color, 1}, std::nullopt, [&](const render::PassContext& ctx) {
                const VkViewport viewport{
                    .x = 0.0F,
                    .y = 0.0F,
                    .width = static_cast<float>(ctx.extent.width),
                    .height = static_cast<float>(ctx.extent.height),
                    .minDepth = 0.0F,
                    .maxDepth = 1.0F,
                };
                const VkRect2D scissor{{0, 0}, ctx.extent};
                vkCmdSetViewport(ctx.cmd, 0, 1, &viewport);
                vkCmdSetScissor(ctx.cmd, 0, 1, &scissor);
                vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe->handle());
                vkCmdDraw(ctx.cmd, 3, 1, 0, 0);
            });
        fg.present(backbuffer);

        auto compiled = fg.compile(*alloc);
        REQUIRE_MESSAGE(compiled.has_value(), compiled.error().message());
        fg.execute(*cmd);
        REQUIRE(vkEndCommandBuffer(*cmd) == VK_SUCCESS);

        const VkSemaphoreSubmitInfo wait_si{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = acquire_sem->handle(),
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        };
        const VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = *cmd,
        };
        const VkSemaphoreSubmitInfo signal_si{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = render_sems[image_index].handle(),
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        const VkSubmitInfo2 submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &wait_si,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_si,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos = &signal_si,
        };
        REQUIRE(vkQueueSubmit2(dev->graphics_queue(), 1, &submit, fence->handle()) ==
                VK_SUCCESS);
        REQUIRE(fence->wait(5s));
        fence->reset();

        const VkSwapchainKHR swapchain = sc->handle();
        const VkSemaphore render_done = render_sems[image_index].handle();
        const VkPresentInfoKHR present{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_done,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
        };
        const VkResult pr = vkQueuePresentKHR(dev->graphics_queue(), &present);
        CHECK((pr == VK_SUCCESS || pr == VK_SUBOPTIMAL_KHR));
    }

    // Teardown order: device idle, then swapchain before surface (§5.4 /
    // VUID-vkDestroySurfaceKHR-surface-01266).
    vkDeviceWaitIdle(dev->handle());
    sc = {};
    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}

TEST_CASE("Buffer: host-visible write + device-local alloc via VMA" *
          doctest::skip(!vulkan_available())) {
    auto inst = gpu::Instance::create({});
    REQUIRE(inst.has_value());
    auto dev = gpu::Device::create(*inst, {});
    REQUIRE_MESSAGE(dev.has_value(), dev.error().message());
    auto alloc = gpu::Allocator::create(*inst, *dev);
    REQUIRE(alloc.has_value());

    auto staging = gpu::Buffer::create(*alloc,
                                       {.size = 256,
                                        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        .host_visible = true,
                                        .debug_name = "staging_test"});
    REQUIRE_MESSAGE(staging.has_value(), staging.error().message());
    CHECK(staging->mapped() != nullptr);
    const std::array<std::byte, 4> payload{std::byte{1}, std::byte{2}, std::byte{3},
                                           std::byte{4}};
    CHECK(staging->write(payload));
    CHECK(!staging->write(std::span<const std::byte>(payload), 4096)); // out of bounds

    auto device_local = gpu::Buffer::create(
        *alloc, {.size = 1024 * 1024,
                 .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 .debug_name = "device_local_test"});
    REQUIRE(device_local.has_value());
    CHECK(device_local->mapped() == nullptr);

    const gpu::BudgetStats budget = alloc->device_local_budget();
    CHECK(budget.budget_bytes > 0);
    MESSAGE("VRAM budget: ", budget.budget_bytes / (1024 * 1024), " MiB, used: ",
            budget.usage_bytes / (1024 * 1024), " MiB");
}
