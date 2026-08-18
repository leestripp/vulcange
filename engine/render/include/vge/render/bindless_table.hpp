#pragma once

// vge::render::BindlessTable — global bindless descriptor set (Plan §5.3).
// Fallback path per §5.6: VK_EXT_descriptor_indexing (core in 1.2) with a
// single persistent set; VK_EXT_descriptor_buffer is the primary-path upgrade.
//
// Layout (set 0):
//   binding 0: uniform buffer  — per-frame constants (view/proj, lights)
//   binding 1: storage buffer  — vertex pulling / global geometry data
//   binding 2: combined image sampler[N] — global texture table, indexed by
//              material push constants with nonuniformEXT
//
// The texture table is update-after-bind + partially-bound + variable-count:
// indices are handed out linearly each frame; registering overwrites the slot.

#include <cstdint>
#include <expected>

#include <vulkan/vulkan.h>

#include <vge/gpu/device.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::render {

class BindlessTable {
public:
    static constexpr std::uint32_t frame_ubo_binding = 0;
    static constexpr std::uint32_t storage_binding = 1;
    static constexpr std::uint32_t textures_binding = 2;
    static constexpr std::uint32_t default_max_textures = 1024;

    BindlessTable() = default;
    ~BindlessTable();
    BindlessTable(const BindlessTable&) = delete;
    BindlessTable& operator=(const BindlessTable&) = delete;
    BindlessTable(BindlessTable&& o) noexcept;
    BindlessTable& operator=(BindlessTable&& o) noexcept;

    [[nodiscard]] static std::expected<BindlessTable, gpu::Error>
    create(const gpu::Device& device,
           std::uint32_t max_textures = default_max_textures);

    [[nodiscard]] VkDescriptorSetLayout layout() const noexcept { return layout_; }
    [[nodiscard]] VkDescriptorSet set() const noexcept { return set_; }

    void bind_frame_ubo(VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset = 0) const;
    void bind_storage_buffer(VkBuffer buffer, VkDeviceSize size,
                             VkDeviceSize offset = 0) const;

    // Registers a texture into the next table slot; returns its index for
    // push-constant material indexing (§5.3).
    [[nodiscard]] std::uint32_t register_texture(VkImageView view, VkSampler sampler);

    // Rewinds the linear index allocator (call once per frame before
    // re-registering this frame's textures).
    void reset_textures() noexcept { next_texture_ = 0; }

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    std::uint32_t max_textures_ = 0;
    std::uint32_t next_texture_ = 0;

    void destroy() noexcept;
};

} // namespace vulcanGE::render
