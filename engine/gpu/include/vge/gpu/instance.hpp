#pragma once

// vge::gpu::Instance — VkInstance + validation + debug utils (Plan §16.3).
// KHRONOS_validation in dev builds; VK_EXT_debug_utils messenger routes to a
// user callback (stderr by default; engine log system per §16.5 plugs in later).

#include <expected>
#include <functional>
#include <span>
#include <string>
#include <string_view>

#include <vulkan/vulkan.h>

#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

using DebugCallback =
    std::function<void(VkDebugUtilsMessageSeverityFlagBitsEXT severity, std::string_view message)>;

struct InstanceDesc {
    std::string_view app_name = "vulcanGE";
    std::uint32_t app_version = VK_MAKE_API_VERSION(0, 0, 1, 0);
    // Validation defaults ON in non-NDEBUG builds (Plan §16.3).
#ifdef NDEBUG
    bool enable_validation = false;
#else
    bool enable_validation = true;
#endif
    std::span<const char* const> required_extensions; // e.g. from GLFW
    DebugCallback debug_callback; // optional; stderr if empty
};

class Instance {
public:
    Instance() = default;
    ~Instance();
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&& o) noexcept;
    Instance& operator=(Instance&& o) noexcept;

    [[nodiscard]] static std::expected<Instance, Error> create(const InstanceDesc& desc = {});

    [[nodiscard]] VkInstance handle() const noexcept { return instance_; }
    [[nodiscard]] bool validation_enabled() const noexcept { return validation_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    bool validation_ = false;
};

} // namespace vulcanGE::gpu
