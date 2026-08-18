#pragma once

// vge::platform::Window — GLFW window on Wayland, Vulkan-surface aware.
// GLFW is the locked windowing/input backend (Plan §2 / directives).

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

#include <vulkan/vulkan.h>

struct GLFWwindow;

namespace vulcanGE::platform {

enum class WindowError {
    glfw_init_failed,
    wayland_unavailable,
    window_creation_failed,
    vulkan_not_available_in_glfw,
    surface_creation_failed,
};

[[nodiscard]] std::string_view to_string(WindowError e) noexcept;

struct WindowDesc {
    std::string_view title = "vulcanGE";
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    bool resizable = true;
};

class Window {
public:
    Window() = default;
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&& o) noexcept;
    Window& operator=(Window&& o) noexcept;

    [[nodiscard]] static std::expected<Window, WindowError> create(const WindowDesc& desc = {});

    // Instance extensions GLFW needs (wayland_surface etc.). Valid before create().
    [[nodiscard]] static std::expected<std::span<const char* const>, WindowError>
    required_instance_extensions();

    [[nodiscard]] std::expected<VkSurfaceKHR, WindowError>
    create_surface(VkInstance instance) const;

    [[nodiscard]] GLFWwindow* handle() const noexcept { return handle_; }
    [[nodiscard]] bool should_close() const;
    [[nodiscard]] VkExtent2D framebuffer_extent() const;

    static void poll_events();

private:
    GLFWwindow* handle_ = nullptr;

    // Process-wide GLFW lifetime, refcounted by live Windows.
    static int live_windows_;
    static bool glfw_initialized_;
    [[nodiscard]] static std::expected<void, WindowError> glfw_acquire();
    static void glfw_release() noexcept;
};

} // namespace vulcanGE::platform
