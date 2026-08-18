#pragma once

// vge::gpu::Allocator — VulkanMemoryAllocator integration.
// Plan §16: VRAM budgets + fragmentation tracking via VMA statistics.
// Plan §12.1: staging buffers for the io_uring -> staging -> GPU streaming path.
//
// Lifetime: create after Device, destroy before it. Buffer/Image allocate
// through this object.

#include <cstdint>
#include <expected>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <vge/gpu/device.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

// Per-heap budget snapshot (vmaGetHeapBudgets). Plan §12.2 memory budgets.
struct BudgetStats {
    std::uint64_t budget_bytes = 0;     // usable heap budget (VK_EXT_memory_budget-aware)
    std::uint64_t usage_bytes = 0;      // current VMA usage of the heap
    std::uint64_t allocation_bytes = 0; // live allocation payload bytes
    std::uint64_t block_bytes = 0;      // VkDeviceMemory block bytes held
};

class Allocator {
public:
    Allocator() = default;
    ~Allocator();
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&& o) noexcept;
    Allocator& operator=(Allocator&& o) noexcept;

    [[nodiscard]] static std::expected<Allocator, Error>
    create(const Instance& instance, const Device& device);

    [[nodiscard]] VmaAllocator handle() const noexcept { return allocator_; }
    [[nodiscard]] const Device& device() const noexcept { return *device_; }

    // Device-local heap budget (Plan §12.2: over-budget triggers LRU eviction,
    // never allocation failure).
    [[nodiscard]] BudgetStats device_local_budget() const;

private:
    VmaAllocator allocator_ = nullptr;
    const Device* device_ = nullptr; // borrowed; outlives us

    void destroy() noexcept;
};

} // namespace vulcanGE::gpu
