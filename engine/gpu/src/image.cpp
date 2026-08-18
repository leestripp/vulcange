#include <vge/gpu/image.hpp>

namespace vulcanGE::gpu {

bool Image::is_depth_format(VkFormat f) noexcept {
    switch (f) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

bool Image::is_stencil_format(VkFormat f) noexcept {
    switch (f) {
    case VK_FORMAT_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

namespace {

[[nodiscard]] VkImageAspectFlags derive_aspect(VkFormat f) noexcept {
    VkImageAspectFlags a = 0;
    if (Image::is_depth_format(f)) {
        a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (Image::is_stencil_format(f)) {
        a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return a != 0 ? a : VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT};
}

} // namespace

std::expected<Image, Error> Image::create(const Allocator& allocator,
                                          const ImageDesc& desc) {
    const VkImageCreateInfo image_ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = desc.depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        .format = desc.format,
        .extent = {desc.width, desc.height, desc.depth},
        .mipLevels = desc.mip_levels,
        .arrayLayers = desc.array_layers,
        .samples = desc.samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = desc.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo alloc_ci{
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    Image out;
    if (const VkResult r = vmaCreateImage(allocator.handle(), &image_ci, &alloc_ci,
                                          &out.image_, &out.allocation_, nullptr);
        r != VK_SUCCESS) {
        return std::unexpected(
            make_error(GpuError::image_creation_failed, r, "vmaCreateImage failed"));
    }
    out.allocator_ = allocator.handle();
    out.device_ = allocator.device().handle();
    out.format_ = desc.format;
    out.extent_ = {desc.width, desc.height, desc.depth};
    out.aspect_ = desc.aspect != 0 ? desc.aspect : derive_aspect(desc.format);

    const VkImageViewCreateInfo view_ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = out.image_,
        .viewType = desc.view_type,
        .format = desc.format,
        .subresourceRange = {out.aspect_, 0, desc.mip_levels, 0, desc.array_layers},
    };
    if (const VkResult r =
            vkCreateImageView(out.device_, &view_ci, nullptr, &out.view_);
        r != VK_SUCCESS) {
        out.destroy();
        return std::unexpected(
            make_error(GpuError::image_view_creation_failed, r, "vkCreateImageView failed"));
    }
    if (!desc.debug_name.empty()) {
        allocator.device().set_object_name(VK_OBJECT_TYPE_IMAGE,
                                           reinterpret_cast<std::uint64_t>(out.image_),
                                           desc.debug_name);
        allocator.device().set_object_name(VK_OBJECT_TYPE_IMAGE_VIEW,
                                           reinterpret_cast<std::uint64_t>(out.view_),
                                           desc.debug_name + "_view");
    }
    return out;
}

void Image::destroy() noexcept {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    if (image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = nullptr;
    }
    allocator_ = nullptr;
    device_ = VK_NULL_HANDLE;
}

Image::~Image() { destroy(); }

Image::Image(Image&& o) noexcept
    : allocator_(o.allocator_), allocation_(o.allocation_), device_(o.device_),
      image_(o.image_), view_(o.view_), format_(o.format_), extent_(o.extent_),
      aspect_(o.aspect_) {
    o.image_ = VK_NULL_HANDLE;
    o.view_ = VK_NULL_HANDLE;
    o.allocation_ = nullptr;
    o.allocator_ = nullptr;
    o.device_ = VK_NULL_HANDLE;
}

Image& Image::operator=(Image&& o) noexcept {
    if (this != &o) {
        destroy();
        allocator_ = o.allocator_;
        allocation_ = o.allocation_;
        device_ = o.device_;
        image_ = o.image_;
        view_ = o.view_;
        format_ = o.format_;
        extent_ = o.extent_;
        aspect_ = o.aspect_;
        o.image_ = VK_NULL_HANDLE;
        o.view_ = VK_NULL_HANDLE;
        o.allocation_ = nullptr;
        o.allocator_ = nullptr;
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vulcanGE::gpu
