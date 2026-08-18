#pragma once

// vge::gpu sync primitives — fences and binary/timeline semaphores.
// Plan §5.4: per-frame fences + semaphore recycling for frames in flight.

#include <chrono>
#include <cstdint>
#include <expected>

#include <vulkan/vulkan.h>

#include <vge/gpu/device.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

class Fence {
public:
    Fence() = default;
    ~Fence();
    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;
    Fence(Fence&& o) noexcept;
    Fence& operator=(Fence&& o) noexcept;

    [[nodiscard]] static std::expected<Fence, Error> create(const Device& device,
                                                            bool signaled = false);

    [[nodiscard]] VkFence handle() const noexcept { return fence_; }

    // Returns true when signaled (false on timeout).
    [[nodiscard]] bool wait(std::chrono::nanoseconds timeout) const;
    void reset() const;

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkFence fence_ = VK_NULL_HANDLE;

    void destroy() noexcept;
};

class Semaphore {
public:
    Semaphore() = default;
    ~Semaphore();
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    Semaphore(Semaphore&& o) noexcept;
    Semaphore& operator=(Semaphore&& o) noexcept;

    // Binary semaphore (swapchain acquire/present sync).
    [[nodiscard]] static std::expected<Semaphore, Error> create(const Device& device);
    // Timeline semaphore with an initial counter value (async compute/transfer
    // overlap, Plan §5.5).
    [[nodiscard]] static std::expected<Semaphore, Error>
    create_timeline(const Device& device, std::uint64_t initial_value = 0);

    [[nodiscard]] VkSemaphore handle() const noexcept { return sem_; }
    [[nodiscard]] bool is_timeline() const noexcept { return timeline_; }

    // Timeline only: current counter value / host wait / host signal.
    [[nodiscard]] std::uint64_t value() const;
    [[nodiscard]] bool wait(std::uint64_t value, std::chrono::nanoseconds timeout) const;
    void signal(std::uint64_t value) const;

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkSemaphore sem_ = VK_NULL_HANDLE;
    bool timeline_ = false;

    void destroy() noexcept;
};

} // namespace vulcanGE::gpu
