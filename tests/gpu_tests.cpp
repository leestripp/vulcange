// GPU bootstrap tests — run on the dev machine's real driver (RX 7800 XT /
// RADV). Skip gracefully if no Vulkan driver is present (e.g. VM without GPU).

#include <doctest/doctest.h>

#include <vge/gpu/gpu.hpp>
#include <vge/platform/window.hpp>

using namespace vulcanGE;

namespace {

[[nodiscard]] bool vulkan_available() {
    return gpu::Instance::create({.enable_validation = false}).has_value();
}

} // namespace

TEST_CASE("Instance: creates with validation in dev builds" * doctest::skip(!vulkan_available())) {
    auto inst = gpu::Instance::create({});
    REQUIRE(inst.has_value());
    CHECK(inst->handle() != VK_NULL_HANDLE);
    CHECK(inst->validation_enabled()); // debug build → validation on (§16.3)
}

TEST_CASE("Instance: moves without double-destroy" * doctest::skip(!vulkan_available())) {
    auto inst = gpu::Instance::create({});
    REQUIRE(inst.has_value());
    const VkInstance h = inst->handle();
    gpu::Instance moved = std::move(*inst);
    CHECK(inst->handle() == VK_NULL_HANDLE);
    CHECK(moved.handle() == h);
}

TEST_CASE("Device: selects GPU meeting min spec (§5.6)" * doctest::skip(!vulkan_available())) {
    auto inst = gpu::Instance::create({});
    REQUIRE(inst.has_value());
    auto dev = gpu::Device::create(*inst, {}); // headless: no surface
    REQUIRE_MESSAGE(dev.has_value(), dev.error().message());

    const auto& info = dev->info();
    MESSAGE("GPU: ", info.name);
    CHECK(info.api_version >= VK_API_VERSION_1_3);
    CHECK(info.dedicated_vram_bytes >= 8ull * 1024 * 1024 * 1024);
    CHECK(info.queues.graphics != ~0u);
    CHECK(dev->graphics_queue() != VK_NULL_HANDLE);

    // Feature matrix is queried for §5.6 fallback paths.
    MESSAGE("mesh_shader=", info.features.mesh_shader,
            " descriptor_buffer=", info.features.descriptor_buffer,
            " rt=", info.features.ray_tracing_pipeline);
    CHECK(info.features.descriptor_indexing); // universal on min-spec GPUs
}

TEST_CASE("Device: rejects fictional below-spec requirement" * doctest::skip(!vulkan_available())) {
    auto inst = gpu::Instance::create({});
    REQUIRE(inst.has_value());
    // Require an absurd 1 TiB of VRAM — must fail with no_suitable_physical_device.
    auto dev = gpu::Device::create(
        *inst, {.min_dedicated_vram = 1ull << 40, .allow_below_min_spec = false});
    CHECK(!dev.has_value());
    if (!dev) {
        CHECK(dev.error().code == gpu::GpuError::no_suitable_physical_device);
    }
    // Same query with override must succeed.
    auto dev2 = gpu::Device::create(
        *inst, {.min_dedicated_vram = 1ull << 40, .allow_below_min_spec = true});
    CHECK(dev2.has_value());
}

TEST_CASE("Window: creates on Wayland and reports extent" * doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_test", .width = 320, .height = 240});
    REQUIRE_MESSAGE(win.has_value(), platform::to_string(win.error()));
    const VkExtent2D ext = win->framebuffer_extent();
    CHECK(ext.width > 0);
    CHECK(ext.height > 0);

    auto exts = platform::Window::required_instance_extensions();
    REQUIRE(exts.has_value());
    CHECK(exts->size() >= 2); // VK_KHR_surface + VK_KHR_wayland_surface
}

TEST_CASE("Swapchain: full chain on Wayland surface" * doctest::skip(!vulkan_available())) {
    auto win = platform::Window::create({.title = "vge_sc_test", .width = 320, .height = 240});
    REQUIRE(win.has_value());
    auto exts = platform::Window::required_instance_extensions();
    REQUIRE(exts.has_value());
    auto inst = gpu::Instance::create({.required_extensions = *exts});
    REQUIRE(inst.has_value());
    auto surface = win->create_surface(inst->handle());
    REQUIRE(surface.has_value());

    auto dev = gpu::Device::create(*inst, {.surface = *surface});
    REQUIRE_MESSAGE(dev.has_value(), dev.error().message());

    // Teardown order (VUID-vkDestroySurfaceKHR-surface-01266): ALL swapchains
    // must be destroyed before their surface, so scope them here. Locals then
    // unwind in order: device, surface, instance, window.
    {
        auto sc = gpu::Swapchain::create(*dev, *surface, {.extent = {320, 240}});
        REQUIRE_MESSAGE(sc.has_value(), sc.error().message());
        CHECK(sc->image_count() >= 2);
        CHECK(sc->extent().width > 0);
        CHECK(sc->format() != VK_FORMAT_UNDEFINED);
        CHECK(sc->image_view(0) != VK_NULL_HANDLE);

        // Recreation via old-swapchain path (resize handling, §5.4).
        auto sc2 = gpu::Swapchain::create(*dev, *surface, {.extent = {320, 240}}, sc->handle());
        REQUIRE(sc2.has_value());
        CHECK(sc2->handle() != VK_NULL_HANDLE);
    }

    vkDestroySurfaceKHR(inst->handle(), *surface, nullptr);
}
