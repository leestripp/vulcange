#include <vge/gpu/swapchain.hpp>

#include <algorithm>

namespace vulcanGE::gpu {

namespace {

[[nodiscard]] std::optional<VkSurfaceFormatKHR>
choose_format(VkPhysicalDevice pd, VkSurfaceKHR surface, VkFormat want_format,
              VkColorSpaceKHR want_space) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, formats.data());
    if (formats.empty()) {
        return std::nullopt;
    }
    // Exact match first, then any-SRGB fallback, then whatever the driver gives.
    for (const auto& f : formats) {
        if (f.format == want_format && f.colorSpace == want_space) {
            return f;
        }
    }
    for (const auto& f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.front();
}

[[nodiscard]] std::optional<VkPresentModeKHR>
choose_present_mode(VkPhysicalDevice pd, VkSurfaceKHR surface, VkPresentModeKHR want) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, modes.data());
    if (modes.empty()) {
        return std::nullopt;
    }
    if (std::find(modes.begin(), modes.end(), want) != modes.end()) {
        return want;
    }
    // FIFO is guaranteed by the spec; if absent something is deeply wrong but
    // take the first mode rather than fail.
    return modes.front();
}

} // namespace

std::expected<Swapchain, Error>
Swapchain::create(const Device& device, VkSurfaceKHR surface, const SwapchainDesc& desc,
                  VkSwapchainKHR old) {
    const VkPhysicalDevice pd = device.physical_handle();

    VkSurfaceCapabilitiesKHR caps{};
    if (const VkResult r = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps);
        r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::swapchain_creation_failed, r,
                                          "surface capabilities query failed"));
    }

    const auto format = choose_format(pd, surface, desc.format, desc.color_space);
    if (!format) {
        return std::unexpected(
            make_error(GpuError::no_surface_format_available, VK_ERROR_FORMAT_NOT_SUPPORTED));
    }
    const auto mode = choose_present_mode(pd, surface, desc.present_mode);
    if (!mode) {
        return std::unexpected(
            make_error(GpuError::no_present_mode_available, VK_ERROR_FEATURE_NOT_PRESENT));
    }

    VkExtent2D extent = desc.extent;
    if (extent.width == 0 || extent.height == 0) {
        if (caps.currentExtent.width != 0xFFFFFFFFu) {
            extent = caps.currentExtent; // Wayland: compositor-driven
        } else {
            extent.width = std::clamp(extent.width, caps.minImageExtent.width,
                                      caps.maxImageExtent.width);
            extent.height = std::clamp(extent.height, caps.minImageExtent.height,
                                       caps.maxImageExtent.height);
        }
    }
    if (extent.width == 0 || extent.height == 0) {
        // Minimized window — caller should skip frame, retry next poll.
        return std::unexpected(
            make_error(GpuError::swapchain_creation_failed, VK_ERROR_SURFACE_LOST_KHR,
                       "zero-area extent (minimized?)"));
    }

    std::uint32_t image_count = std::max(desc.min_image_count, caps.minImageCount);
    if (caps.maxImageCount > 0) {
        image_count = std::min(image_count, caps.maxImageCount);
    }

    const auto& q = device.info().queues;
    const std::uint32_t queue_indices[] = {q.graphics, q.compute};
    const bool concurrent = q.graphics != q.compute;

    VkSwapchainCreateInfoKHR ci{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format->format,
        .imageColorSpace = format->colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = concurrent ? 2u : 0u,
        .pQueueFamilyIndices = concurrent ? queue_indices : nullptr,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = *mode,
        .clipped = VK_TRUE,
        .oldSwapchain = old,
    };

    Swapchain out;
    out.device_ = device.handle();
    if (const VkResult r = vkCreateSwapchainKHR(out.device_, &ci, nullptr, &out.swapchain_);
        r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::swapchain_creation_failed, r));
    }
    out.format_ = format->format;
    out.extent_ = extent;

    std::uint32_t actual = 0;
    vkGetSwapchainImagesKHR(out.device_, out.swapchain_, &actual, nullptr);
    out.images_.resize(actual);
    vkGetSwapchainImagesKHR(out.device_, out.swapchain_, &actual, out.images_.data());

    out.views_.reserve(actual);
    for (const VkImage img : out.images_) {
        VkImageViewCreateInfo view_ci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format->format,
            .components = {},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VkImageView view = VK_NULL_HANDLE;
        if (const VkResult r = vkCreateImageView(out.device_, &view_ci, nullptr, &view);
            r != VK_SUCCESS) {
            out.destroy();
            return std::unexpected(make_error(GpuError::swapchain_creation_failed, r,
                                              "image view creation failed"));
        }
        out.views_.push_back(view);
    }
    return out;
}

void Swapchain::destroy() noexcept {
    for (const VkImageView v : views_) {
        vkDestroyImageView(device_, v, nullptr);
    }
    views_.clear();
    images_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

Swapchain::~Swapchain() { destroy(); }

Swapchain::Swapchain(Swapchain&& o) noexcept
    : device_(o.device_), swapchain_(o.swapchain_), format_(o.format_), extent_(o.extent_),
      images_(std::move(o.images_)), views_(std::move(o.views_)) {
    o.swapchain_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
}

Swapchain& Swapchain::operator=(Swapchain&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        swapchain_ = o.swapchain_;
        format_ = o.format_;
        extent_ = o.extent_;
        images_ = std::move(o.images_);
        views_ = std::move(o.views_);
        o.swapchain_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vulcanGE::gpu
