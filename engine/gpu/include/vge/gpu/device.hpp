#pragma once

// vge::gpu::Device — physical device selection per Plan §5.6 feature matrix,
// logical device + queue families per Plan §5.5.

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <vge/gpu/error.hpp>
#include <vge/gpu/instance.hpp>

namespace vulcanGE::gpu {

// Plan §5.6 — queried support; primary/fallback paths chosen by subsystems.
struct GpuFeatureMatrix {
    bool mesh_shader = false;          // VK_EXT_mesh_shader (primary path)
    bool descriptor_buffer = false;    // VK_EXT_descriptor_buffer (bindless primary)
    bool descriptor_indexing = false;  // VK_EXT_descriptor_indexing (bindless fallback)
    bool ray_tracing_pipeline = false; // VK_KHR_ray_tracing_pipeline (optional, §6.6)
    bool present_id = false;           // VK_KHR_present_id (pacing telemetry, §5.4)
    bool present_wait = false;         // VK_KHR_present_wait
    bool memory_budget = false;        // VK_EXT_memory_budget
};

// Plan §5.5 — graphics required; dedicated compute/transfer preferred.
struct QueueFamilies {
    std::uint32_t graphics = ~0u;
    std::uint32_t compute = ~0u;  // dedicated if possible, else == graphics
    std::uint32_t transfer = ~0u; // dedicated if possible, else == graphics
    bool dedicated_compute = false;
    bool dedicated_transfer = false;
};

struct GpuInfo {
    std::string name;
    VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    std::uint32_t api_version = 0;
    std::uint64_t dedicated_vram_bytes = 0;
    GpuFeatureMatrix features;
    QueueFamilies queues;
    VkPhysicalDeviceLimits limits{};
};

struct DeviceDesc {
    // 8 GiB per Plan §5.6 min spec; override for dev/CI VMs.
    std::uint64_t min_dedicated_vram = 8ull * 1024 * 1024 * 1024;
    bool allow_below_min_spec = false;
    VkSurfaceKHR surface = VK_NULL_HANDLE; // null = headless (no present requirement)
};

class Device {
public:
    Device() = default;
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& o) noexcept;
    Device& operator=(Device&& o) noexcept;

    [[nodiscard]] static std::expected<Device, Error>
    create(const Instance& instance, const DeviceDesc& desc = {});

    [[nodiscard]] VkDevice handle() const noexcept { return device_; }
    [[nodiscard]] VkPhysicalDevice physical_handle() const noexcept { return physical_; }
    [[nodiscard]] const GpuInfo& info() const noexcept { return info_; }

    [[nodiscard]] VkQueue graphics_queue() const noexcept { return graphics_queue_; }
    [[nodiscard]] VkQueue compute_queue() const noexcept { return compute_queue_; }
    [[nodiscard]] VkQueue transfer_queue() const noexcept { return transfer_queue_; }

    // Debug name via VK_EXT_debug_utils (Plan §16.3); no-op without validation.
    void set_object_name(VkObjectType type, std::uint64_t object, std::string_view name) const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkQueue transfer_queue_ = VK_NULL_HANDLE;
    GpuInfo info_;
};

} // namespace vulcanGE::gpu
