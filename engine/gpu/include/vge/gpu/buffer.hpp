#pragma once

// vge::gpu::Buffer — VMA-backed VkBuffer RAII.
// - host_visible=true: persistently mapped upload/staging buffer
//   (VMA HOST_ACCESS_SEQUENTIAL_WRITE + MAPPED); write() flushes for
//   non-coherent memory.
// - host_visible=false: device-local; fill via staging + vkCmdCopyBuffer.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <vge/gpu/allocator.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

struct BufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    bool host_visible = false; // CPU -> GPU upload (sequential write, mapped)
    bool host_readable = false; // GPU -> CPU readback (random access, mapped)
    std::string debug_name;
};

class Buffer {
public:
    Buffer() = default;
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept;
    Buffer& operator=(Buffer&& o) noexcept;

    [[nodiscard]] static std::expected<Buffer, Error>
    create(const Allocator& allocator, const BufferDesc& desc);

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }

    // Non-null only for host_visible buffers (persistently mapped).
    [[nodiscard]] std::byte* mapped() const noexcept { return mapped_; }

    // memcpy + flush (non-coherent safe). No-op returns false when not
    // host-visible or range is out of bounds.
    bool write(std::span<const std::byte> data, VkDeviceSize offset = 0) const;

    // Makes GPU writes visible to subsequent host reads (readback buffers).
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const;

private:
    VmaAllocator allocator_ = nullptr; // borrowed from Allocator (outlives us)
    VmaAllocation allocation_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    std::byte* mapped_ = nullptr;

    void destroy() noexcept;
};

} // namespace vulcanGE::gpu
