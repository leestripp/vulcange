#include <vge/gpu/command.hpp>

namespace vulcanGE::gpu {

std::expected<CommandPool, Error> CommandPool::create(const Device& device,
                                                      std::uint32_t queue_family) {
    const VkCommandPoolCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        // Buffers are re-recorded every frame (Frame Graph, Plan §5.1).
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family,
    };
    CommandPool out;
    if (const VkResult r = vkCreateCommandPool(device.handle(), &ci, nullptr, &out.pool_);
        r != VK_SUCCESS) {
        return std::unexpected(
            make_error(GpuError::command_pool_creation_failed, r, "vkCreateCommandPool failed"));
    }
    out.device_ = device.handle();
    return out;
}

void CommandPool::destroy() noexcept {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

CommandPool::~CommandPool() { destroy(); }

CommandPool::CommandPool(CommandPool&& o) noexcept : device_(o.device_), pool_(o.pool_) {
    o.pool_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
}

CommandPool& CommandPool::operator=(CommandPool&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        pool_ = o.pool_;
        o.pool_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

std::expected<VkCommandBuffer, Error> CommandPool::allocate() const {
    const VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (const VkResult r = vkAllocateCommandBuffers(device_, &ai, &cmd); r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::command_buffer_allocation_failed, r,
                                          "vkAllocateCommandBuffers failed"));
    }
    return cmd;
}

void CommandPool::reset() const {
    vkResetCommandPool(device_, pool_, 0);
}

} // namespace vulcanGE::gpu
