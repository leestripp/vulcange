#pragma once

// vge::gpu::Image — VMA-backed VkImage + full-range VkImageView RAII.
// Used for transient Frame Graph render targets (G-Buffer, Hi-Z, shadow maps)
// and cooked textures. Aspect is derived from the format when left zero.

#include <cstdint>
#include <expected>
#include <string>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <vge/gpu/allocator.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

struct ImageDesc {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t depth = 1;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
    VkImageAspectFlags aspect = 0; // 0 = derive from format (color/depth/stencil)
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    std::string debug_name;
};

class Image {
public:
    Image() = default;
    ~Image();
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& o) noexcept;
    Image& operator=(Image&& o) noexcept;

    [[nodiscard]] static std::expected<Image, Error>
    create(const Allocator& allocator, const ImageDesc& desc);

    [[nodiscard]] VkImage handle() const noexcept { return image_; }
    [[nodiscard]] VkImageView view() const noexcept { return view_; }
    [[nodiscard]] VkFormat format() const noexcept { return format_; }
    [[nodiscard]] VkExtent3D extent() const noexcept { return extent_; }
    [[nodiscard]] VkImageAspectFlags aspect() const noexcept { return aspect_; }

    // Format classification helpers (used by the Frame Graph for layout picks).
    [[nodiscard]] static bool is_depth_format(VkFormat f) noexcept;
    [[nodiscard]] static bool is_stencil_format(VkFormat f) noexcept;

private:
    VmaAllocator allocator_ = nullptr; // borrowed from Allocator (outlives us)
    VmaAllocation allocation_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE; // borrowed (view creation)
    VkImage image_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent3D extent_{};
    VkImageAspectFlags aspect_ = 0;

    void destroy() noexcept;
};

} // namespace vulcanGE::gpu
