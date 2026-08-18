#pragma once

// vge::gpu::CommandPool — primary command buffer allocation for one queue
// family. Plan §4.5: one pool per frame-in-flight per recording thread;
// pools are reset wholesale (vkResetCommandPool) instead of freeing buffers.

#include <cstdint>
#include <expected>

#include <vulkan/vulkan.h>

#include <vge/gpu/device.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

class CommandPool {
public:
    CommandPool() = default;
    ~CommandPool();
    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;
    CommandPool(CommandPool&& o) noexcept;
    CommandPool& operator=(CommandPool&& o) noexcept;

    [[nodiscard]] static std::expected<CommandPool, Error>
    create(const Device& device, std::uint32_t queue_family);

    [[nodiscard]] VkCommandPool handle() const noexcept { return pool_; }

    // Allocated buffers live until reset() or pool destruction.
    [[nodiscard]] std::expected<VkCommandBuffer, Error> allocate() const;

    // Recycles all buffers allocated from this pool.
    void reset() const;

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkCommandPool pool_ = VK_NULL_HANDLE;

    void destroy() noexcept;
};

} // namespace vulcanGE::gpu
