#pragma once

// vge::gpu::Swapchain — per Plan §5.4:
// - FIFO (vsync) default; MAILBOX / IMMEDIATE runtime-selectable via desc.
// - Double-buffered acquire by default (2 images, negotiable up).
// - SDR default (B8G8R8A8_SRGB); HDR formats (HDR10 ST2084 / scRGB) selectable
//   via SwapchainDesc::color_space for the Wayland color-management-v1 path.
// - Recreation driven by acquire/present results (OUT_OF_DATE / SUBOPTIMAL).

#include <cstdint>
#include <expected>
#include <vector>

#include <vulkan/vulkan.h>

#include <vge/gpu/device.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

struct SwapchainDesc {
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR; // §5.4 default
    VkFormat format = VK_FORMAT_B8G8R8A8_SRGB;                // SDR default
    VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    std::uint32_t min_image_count = 2;                        // double-buffered
    VkExtent2D extent = {0, 0}; // {0,0} = derive from surface caps / window
};

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&& o) noexcept;
    Swapchain& operator=(Swapchain&& o) noexcept;

    // Pass the old swapchain for efficient recreation on resize.
    //
    // Lifetime contract: `surface` MUST outlive this Swapchain.
    // VUID-vkDestroySurfaceKHR-surface-01266 requires all swapchains created
    // for a surface to be destroyed before the surface itself; violating this
    // is UB and segfaults in practice (RADV's vkDestroySwapchainKHR flushes
    // the freed Wayland surface). Callers should also ensure the device is
    // idle (no in-flight acquire/present) before destruction.
    [[nodiscard]] static std::expected<Swapchain, Error>
    create(const Device& device, VkSurfaceKHR surface, const SwapchainDesc& desc,
           VkSwapchainKHR old = VK_NULL_HANDLE);

    [[nodiscard]] VkSwapchainKHR handle() const noexcept { return swapchain_; }
    [[nodiscard]] VkFormat format() const noexcept { return format_; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }
    [[nodiscard]] std::uint32_t image_count() const noexcept {
        return static_cast<std::uint32_t>(images_.size());
    }
    [[nodiscard]] VkImage image(std::uint32_t i) const { return images_.at(i); }
    [[nodiscard]] VkImageView image_view(std::uint32_t i) const { return views_.at(i); }

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;

    void destroy() noexcept;
};

} // namespace vulcanGE::gpu
