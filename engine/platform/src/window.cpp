#include <vge/platform/window.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace vulcanGE::platform {

int Window::live_windows_ = 0;
bool Window::glfw_initialized_ = false;

std::string_view to_string(WindowError e) noexcept {
    switch (e) {
    case WindowError::glfw_init_failed: return "GLFW initialization failed";
    case WindowError::wayland_unavailable: return "Wayland platform unavailable in GLFW";
    case WindowError::window_creation_failed: return "GLFW window creation failed";
    case WindowError::vulkan_not_available_in_glfw: return "GLFW reports no Vulkan support";
    case WindowError::surface_creation_failed: return "Vulkan surface creation failed";
    }
    return "unknown window error";
}

std::expected<void, WindowError> Window::glfw_acquire() {
    if (glfw_initialized_) {
        ++live_windows_;
        return {};
    }
    // Wayland-first; GLFW falls back to X11 if the compositor socket is absent.
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    if (glfwInit() != GLFW_TRUE) {
        return std::unexpected(WindowError::glfw_init_failed);
    }
    if (glfwGetPlatform() != GLFW_PLATFORM_WAYLAND) {
        glfwTerminate();
        return std::unexpected(WindowError::wayland_unavailable);
    }
    glfw_initialized_ = true;
    live_windows_ = 1;
    return {};
}

void Window::glfw_release() noexcept {
    if (--live_windows_ <= 0 && glfw_initialized_) {
        glfwTerminate();
        glfw_initialized_ = false;
        live_windows_ = 0;
    }
}

std::expected<std::span<const char* const>, WindowError>
Window::required_instance_extensions() {
    // glfwGetRequiredInstanceExtensions needs GLFW initialized.
    const bool temp_init = !glfw_initialized_;
    if (temp_init && glfwInit() != GLFW_TRUE) {
        return std::unexpected(WindowError::glfw_init_failed);
    }
    std::uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);
    if (temp_init) {
        glfwTerminate();
    }
    if (exts == nullptr || count == 0) {
        return std::unexpected(WindowError::vulkan_not_available_in_glfw);
    }
    return std::span<const char* const>{exts, count}; // GLFW-owned, static lifetime
}

std::expected<Window, WindowError> Window::create(const WindowDesc& desc) {
    if (auto r = glfw_acquire(); !r) {
        return std::unexpected(r.error());
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, desc.resizable ? GLFW_TRUE : GLFW_FALSE);
    GLFWwindow* w = glfwCreateWindow(
        static_cast<int>(desc.width), static_cast<int>(desc.height),
        std::string(desc.title).c_str(), nullptr, nullptr);
    if (w == nullptr) {
        glfw_release();
        return std::unexpected(WindowError::window_creation_failed);
    }
    Window win;
    win.handle_ = w;
    return win;
}

Window::~Window() {
    if (handle_ != nullptr) {
        glfwDestroyWindow(handle_);
        glfw_release();
    }
}

Window::Window(Window&& o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

Window& Window::operator=(Window&& o) noexcept {
    if (this != &o) {
        if (handle_ != nullptr) {
            glfwDestroyWindow(handle_);
            glfw_release();
        }
        handle_ = o.handle_;
        o.handle_ = nullptr;
    }
    return *this;
}

std::expected<VkSurfaceKHR, WindowError> Window::create_surface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, handle_, nullptr, &surface) != VK_SUCCESS) {
        return std::unexpected(WindowError::surface_creation_failed);
    }
    return surface;
}

bool Window::should_close() const { return glfwWindowShouldClose(handle_) == GLFW_TRUE; }

VkExtent2D Window::framebuffer_extent() const {
    int w = 0, h = 0;
    glfwGetFramebufferSize(handle_, &w, &h);
    return {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
}

void Window::poll_events() { glfwPollEvents(); }

} // namespace vulcanGE::platform
