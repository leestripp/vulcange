#include <vge/gpu/sync.hpp>

namespace vulcanGE::gpu {

// ---- Fence ----

std::expected<Fence, Error> Fence::create(const Device& device, bool signaled) {
    const VkFenceCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = signaled ? VkFenceCreateFlags{VK_FENCE_CREATE_SIGNALED_BIT}
                          : VkFenceCreateFlags{0},
    };
    Fence out;
    if (const VkResult r = vkCreateFence(device.handle(), &ci, nullptr, &out.fence_);
        r != VK_SUCCESS) {
        return std::unexpected(
            make_error(GpuError::sync_object_creation_failed, r, "vkCreateFence failed"));
    }
    out.device_ = device.handle();
    return out;
}

void Fence::destroy() noexcept {
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

Fence::~Fence() { destroy(); }

Fence::Fence(Fence&& o) noexcept : device_(o.device_), fence_(o.fence_) {
    o.fence_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
}

Fence& Fence::operator=(Fence&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        fence_ = o.fence_;
        o.fence_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

bool Fence::wait(std::chrono::nanoseconds timeout) const {
    const VkResult r =
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, static_cast<std::uint64_t>(timeout.count()));
    return r == VK_SUCCESS;
}

void Fence::reset() const { vkResetFences(device_, 1, &fence_); }

// ---- Semaphore ----

std::expected<Semaphore, Error> Semaphore::create(const Device& device) {
    const VkSemaphoreCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    Semaphore out;
    if (const VkResult r = vkCreateSemaphore(device.handle(), &ci, nullptr, &out.sem_);
        r != VK_SUCCESS) {
        return std::unexpected(
            make_error(GpuError::sync_object_creation_failed, r, "vkCreateSemaphore failed"));
    }
    out.device_ = device.handle();
    out.timeline_ = false;
    return out;
}

std::expected<Semaphore, Error> Semaphore::create_timeline(const Device& device,
                                                           std::uint64_t initial_value) {
    const VkSemaphoreTypeCreateInfo type_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = initial_value,
    };
    const VkSemaphoreCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &type_ci,
    };
    Semaphore out;
    if (const VkResult r = vkCreateSemaphore(device.handle(), &ci, nullptr, &out.sem_);
        r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::sync_object_creation_failed, r,
                                          "vkCreateSemaphore (timeline) failed"));
    }
    out.device_ = device.handle();
    out.timeline_ = true;
    return out;
}

void Semaphore::destroy() noexcept {
    if (sem_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, sem_, nullptr);
        sem_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
    timeline_ = false;
}

Semaphore::~Semaphore() { destroy(); }

Semaphore::Semaphore(Semaphore&& o) noexcept
    : device_(o.device_), sem_(o.sem_), timeline_(o.timeline_) {
    o.sem_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
    o.timeline_ = false;
}

Semaphore& Semaphore::operator=(Semaphore&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        sem_ = o.sem_;
        timeline_ = o.timeline_;
        o.sem_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
        o.timeline_ = false;
    }
    return *this;
}

std::uint64_t Semaphore::value() const {
    std::uint64_t v = 0;
    vkGetSemaphoreCounterValue(device_, sem_, &v);
    return v;
}

bool Semaphore::wait(std::uint64_t value, std::chrono::nanoseconds timeout) const {
    const VkSemaphoreWaitInfo wi{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &sem_,
        .pValues = &value,
    };
    return vkWaitSemaphores(device_, &wi, static_cast<std::uint64_t>(timeout.count())) ==
           VK_SUCCESS;
}

void Semaphore::signal(std::uint64_t value) const {
    const VkSemaphoreSignalInfo si{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = sem_,
        .value = value,
    };
    vkSignalSemaphore(device_, &si);
}

} // namespace vulcanGE::gpu
