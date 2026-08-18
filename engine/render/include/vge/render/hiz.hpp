#pragma once

// vge::render::HiZPyramid — persistent hierarchical-Z depth pyramid for
// two-phase occlusion culling (Plan §5.3). Reverse-Z: stores the MAX depth
// (nearest occluder) per region. mip 0 is half-resolution; each level is the
// 2x2 max of the previous, built by a compute dispatch chain recorded into
// the frame command buffer.
//
// Frame flow (single-phase form used until temporal reprojection lands):
//   depth prepass -> record_build(depth) -> gbuffer task shader culls
//   clusters against the fresh pyramid.

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <vge/gpu/allocator.hpp>
#include <vge/gpu/error.hpp>
#include <vge/gpu/image.hpp>
#include <vge/gpu/pipeline.hpp>

namespace vulcanGE::render {

class BindlessTable;

class HiZPyramid {
public:
    HiZPyramid() = default;
    ~HiZPyramid();
    HiZPyramid(const HiZPyramid&) = delete;
    HiZPyramid& operator=(const HiZPyramid&) = delete;
    HiZPyramid(HiZPyramid&& o) noexcept;
    HiZPyramid& operator=(HiZPyramid&& o) noexcept;

    // Creates a half-res pyramid: mip 0 = (width/2) x (height/2), R32_SFLOAT.
    [[nodiscard]] static std::expected<HiZPyramid, gpu::Error>
    create(const gpu::Device& device, const gpu::Allocator& allocator, std::uint32_t width,
           std::uint32_t height, std::string debug_name = {});

    // Records the full build: samples mip 0 from `depth_view`, then
    // max-downsamples the remaining mips. `depth_view`'s image must already be
    // in a shader-readable layout (GENERAL / SHADER_READ_ONLY) — layout
    // management belongs to the caller (typically a FrameGraph compute pass
    // reading the depth target). Registers all descriptors it needs in
    // `table`. Ends with a compute-write -> task-shader-read barrier so the
    // pyramid is ready for cluster culling in later passes of this frame.
    void record_build(VkCommandBuffer cmd, const gpu::ComputePipeline& pipeline,
                      BindlessTable& table, VkImageView depth_view, VkSampler sampler);

    // Full-mip-chain view for textureLod sampling in the task shader.
    [[nodiscard]] VkImageView full_view() const noexcept { return image_.view(); }
    [[nodiscard]] VkImageView mip_view(std::uint32_t mip) const noexcept { return mip_views_[mip]; }
    [[nodiscard]] std::uint32_t mip_count() const noexcept {
        return static_cast<std::uint32_t>(mip_views_.size());
    }
    [[nodiscard]] VkExtent2D mip_extent(std::uint32_t mip) const noexcept;
    [[nodiscard]] VkImage image() const noexcept { return image_.handle(); }

private:
    gpu::Image image_;
    std::vector<VkImageView> mip_views_; // single-level storage views
    VkDevice device_ = VK_NULL_HANDLE;   // borrowed from Device (outlives us)
    VkExtent2D base_extent_{};
    bool initialized_ = false; // first build transitions UNDEFINED -> GENERAL

    void destroy() noexcept;
};

} // namespace vulcanGE::render
