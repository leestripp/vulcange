#include <vge/gpu/buffer.hpp>

#include <cstring>

namespace vulcanGE::gpu {

std::expected<Buffer, Error> Buffer::create(const Allocator& allocator,
                                            const BufferDesc& desc) {
    if (desc.size == 0) {
        return std::unexpected(
            make_error(GpuError::buffer_creation_failed, VK_ERROR_INITIALIZATION_FAILED,
                       "buffer size must be non-zero"));
    }
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc.size,
        .usage = desc.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateFlags flags = 0;
    if (desc.host_visible) {
        flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    if (desc.host_readable) {
        flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    const VmaAllocationCreateInfo alloc_ci{
        .flags = flags,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    Buffer out;
    VmaAllocationInfo info{};
    if (const VkResult r = vmaCreateBuffer(allocator.handle(), &buffer_ci, &alloc_ci,
                                           &out.buffer_, &out.allocation_, &info);
        r != VK_SUCCESS) {
        return std::unexpected(
            make_error(GpuError::buffer_creation_failed, r, "vmaCreateBuffer failed"));
    }
    out.allocator_ = allocator.handle();
    out.size_ = desc.size;
    out.mapped_ = static_cast<std::byte*>(info.pMappedData);
    if (!desc.debug_name.empty()) {
        allocator.device().set_object_name(VK_OBJECT_TYPE_BUFFER,
                                           reinterpret_cast<std::uint64_t>(out.buffer_),
                                           desc.debug_name);
    }
    return out;
}

void Buffer::destroy() noexcept {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = nullptr;
    }
    allocator_ = nullptr;
    mapped_ = nullptr;
    size_ = 0;
}

Buffer::~Buffer() { destroy(); }

Buffer::Buffer(Buffer&& o) noexcept
    : allocator_(o.allocator_), allocation_(o.allocation_), buffer_(o.buffer_),
      size_(o.size_), mapped_(o.mapped_) {
    o.buffer_ = VK_NULL_HANDLE;
    o.allocation_ = nullptr;
    o.allocator_ = nullptr;
    o.mapped_ = nullptr;
    o.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        destroy();
        allocator_ = o.allocator_;
        allocation_ = o.allocation_;
        buffer_ = o.buffer_;
        size_ = o.size_;
        mapped_ = o.mapped_;
        o.buffer_ = VK_NULL_HANDLE;
        o.allocation_ = nullptr;
        o.allocator_ = nullptr;
        o.mapped_ = nullptr;
        o.size_ = 0;
    }
    return *this;
}

bool Buffer::write(std::span<const std::byte> data, VkDeviceSize offset) const {
    if (mapped_ == nullptr || offset + data.size() > size_) {
        return false;
    }
    std::memcpy(mapped_ + offset, data.data(), data.size());
    // No-op on coherent memory; required for non-coherent HOST_VISIBLE heaps.
    vmaFlushAllocation(allocator_, allocation_, offset, data.size());
    return true;
}

void Buffer::invalidate(VkDeviceSize offset, VkDeviceSize size) const {
    vmaInvalidateAllocation(allocator_, allocation_, offset, size);
}

} // namespace vulcanGE::gpu
