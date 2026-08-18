#include <vge/render/hiz.hpp>

#include <algorithm>

#include <vge/gpu/device.hpp>
#include <vge/render/bindless_table.hpp>

namespace vulcanGE::render {

namespace {

void barrier_image(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags2 src_stage,
                   VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                   VkAccessFlags2 dst_access, VkImageLayout old_layout,
                   VkImageLayout new_layout, std::uint32_t base_mip, std::uint32_t mip_count) {
    const VkImageMemoryBarrier2 b{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_mip, mip_count, 0, 1}};
    const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .imageMemoryBarrierCount = 1,
                               .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cmd, &dep);
}

struct BuildPush {
    std::uint32_t src_tex; // texture table index of the source
    std::uint32_t src_lod; // lod to fetch from the source (0 for the depth copy)
    std::uint32_t dst_img; // storage image table index of the destination mip
    std::uint32_t dst_w;
    std::uint32_t dst_h;
};

} // namespace

std::expected<HiZPyramid, gpu::Error> HiZPyramid::create(const gpu::Device& device,
                                                         const gpu::Allocator& allocator,
                                                         std::uint32_t width, std::uint32_t height,
                                                         std::string debug_name) {
    const std::uint32_t base_w = std::max(1u, width / 2);
    const std::uint32_t base_h = std::max(1u, height / 2);
    std::uint32_t mip_count = 1;
    for (std::uint32_t s = std::max(base_w, base_h); s > 1; s = std::max(1u, s / 2)) {
        ++mip_count;
    }

    auto image = gpu::Image::create(
        allocator, {.width = base_w,
                    .height = base_h,
                    .mip_levels = mip_count,
                    .format = VK_FORMAT_R32_SFLOAT,
                    .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    .debug_name = debug_name.empty() ? "hiz_pyramid" : debug_name});
    if (!image) {
        return std::unexpected(image.error());
    }

    HiZPyramid out;
    out.device_ = device.handle();
    out.base_extent_ = {base_w, base_h};
    out.mip_views_.reserve(mip_count);
    for (std::uint32_t i = 0; i < mip_count; ++i) {
        const VkImageViewCreateInfo view_ci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image->handle(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R32_SFLOAT,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1}};
        VkImageView view = VK_NULL_HANDLE;
        if (const VkResult r = vkCreateImageView(device.handle(), &view_ci, nullptr, &view);
            r != VK_SUCCESS) {
            return std::unexpected(
                gpu::make_error(gpu::GpuError::image_creation_failed, r, "hiz mip view failed"));
        }
        out.mip_views_.push_back(view);
        device.set_object_name(VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<std::uint64_t>(view),
                               (debug_name.empty() ? "hiz" : debug_name) + "_mip" +
                                   std::to_string(i));
    }
    out.image_ = std::move(*image);
    return out;
}

VkExtent2D HiZPyramid::mip_extent(std::uint32_t mip) const noexcept {
    return {std::max(1u, base_extent_.width >> mip), std::max(1u, base_extent_.height >> mip)};
}

void HiZPyramid::record_build(VkCommandBuffer cmd, const gpu::ComputePipeline& pipeline,
                              BindlessTable& table, VkImageView depth_view, VkSampler sampler) {
    // Pyramid: previous frame's task-shader reads (or UNDEFINED) -> GENERAL.
    barrier_image(cmd, image_.handle(),
                  initialized_ ? VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                               : VK_PIPELINE_STAGE_2_NONE,
                  initialized_ ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  initialized_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_GENERAL, 0, mip_count());
    initialized_ = true;

    const std::uint32_t depth_tex = table.register_texture(depth_view, sampler);
    const std::uint32_t pyramid_tex = table.register_texture(image_.view(), sampler);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    const VkDescriptorSet set = table.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &set, 0,
                            nullptr);

    auto dispatch = [&](std::uint32_t mip, std::uint32_t src_tex, std::uint32_t src_lod) {
        const VkExtent2D e = mip_extent(mip);
        const std::uint32_t dst_img = table.register_storage_image(mip_views_[mip]);
        const BuildPush push{src_tex, src_lod, dst_img, e.width, e.height};
        vkCmdPushConstants(cmd, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
        vkCmdDispatch(cmd, (e.width + 7) / 8, (e.height + 7) / 8, 1);
    };

    dispatch(0, depth_tex, 0);

    for (std::uint32_t mip = 1; mip < mip_count(); ++mip) {
        barrier_image(cmd, image_.handle(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                      VK_IMAGE_LAYOUT_GENERAL, mip - 1, 1);
        dispatch(mip, pyramid_tex, mip - 1);
    }

    // Whole chain ready for task-shader culling later this frame.
    barrier_image(cmd, image_.handle(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                  VK_IMAGE_LAYOUT_GENERAL, 0, mip_count());
}

HiZPyramid::~HiZPyramid() { destroy(); }

HiZPyramid::HiZPyramid(HiZPyramid&& o) noexcept
    : image_(std::move(o.image_)),
      mip_views_(std::move(o.mip_views_)),
      device_(o.device_),
      base_extent_(o.base_extent_),
      initialized_(o.initialized_) {
    o.mip_views_.clear();
    o.initialized_ = false;
}

HiZPyramid& HiZPyramid::operator=(HiZPyramid&& o) noexcept {
    if (this != &o) {
        destroy();
        image_ = std::move(o.image_);
        mip_views_ = std::move(o.mip_views_);
        device_ = o.device_;
        base_extent_ = o.base_extent_;
        initialized_ = o.initialized_;
        o.mip_views_.clear();
        o.initialized_ = false;
    }
    return *this;
}

void HiZPyramid::destroy() noexcept {
    if (device_ != VK_NULL_HANDLE) {
        for (VkImageView v : mip_views_) {
            vkDestroyImageView(device_, v, nullptr);
        }
    }
    mip_views_.clear();
}

} // namespace vulcanGE::render
