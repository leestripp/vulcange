#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <vulkan/vulkan.h>

namespace vulcanGE::gpu {

enum class GpuError {
    instance_creation_failed,
    validation_layer_unavailable,
    debug_utils_unavailable,
    no_vulkan_driver,
    no_suitable_physical_device,
    api_version_too_low,
    vram_below_min_spec,
    no_graphics_queue_family,
    device_creation_failed,
    swapchain_support_missing,
    swapchain_creation_failed,
    no_present_mode_available,
    no_surface_format_available,
    allocator_creation_failed,
    buffer_creation_failed,
    image_creation_failed,
    image_view_creation_failed,
    command_pool_creation_failed,
    command_buffer_allocation_failed,
    sync_object_creation_failed,
    shader_load_failed,
    shader_module_creation_failed,
    pipeline_layout_creation_failed,
    pipeline_creation_failed,
};

[[nodiscard]] std::string_view to_string(GpuError e) noexcept;

// Error with Vulkan result context for monadic error handling (Plan §2.1).
struct Error {
    GpuError code;
    VkResult vk_result = VK_SUCCESS;
    std::string detail;

    [[nodiscard]] std::string message() const;
};

[[nodiscard]] inline Error make_error(GpuError code, VkResult res, std::string detail = {}) {
    return Error{code, res, std::move(detail)};
}

} // namespace vulcanGE::gpu
