// VMA implementation TU. Warnings are relaxed here only: vk_mem_alloc.h is
// third-party code that predates our warning policy (vge_warnings stays strict
// for engine code).
#define VMA_IMPLEMENTATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

#include <vge/gpu/allocator.hpp>

namespace vulcanGE::gpu {

std::expected<Allocator, Error> Allocator::create(const Instance& instance,
                                                  const Device& device) {
    const VmaVulkanFunctions functions{
        .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
    };
    const VmaAllocatorCreateInfo ci{
        .flags = device.info().features.memory_budget
                     ? VmaAllocatorCreateFlags{VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT}
                     : VmaAllocatorCreateFlags{0},
        .physicalDevice = device.physical_handle(),
        .device = device.handle(),
        .pVulkanFunctions = &functions,
        .instance = instance.handle(),
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };

    Allocator out;
    if (const VkResult r = vmaCreateAllocator(&ci, &out.allocator_); r != VK_SUCCESS) {
        return std::unexpected(
            make_error(GpuError::allocator_creation_failed, r, "vmaCreateAllocator failed"));
    }
    out.device_ = &device;
    return out;
}

void Allocator::destroy() noexcept {
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
        allocator_ = nullptr;
    }
    device_ = nullptr;
}

Allocator::~Allocator() { destroy(); }

Allocator::Allocator(Allocator&& o) noexcept
    : allocator_(o.allocator_), device_(o.device_) {
    o.allocator_ = nullptr;
    o.device_ = nullptr;
}

Allocator& Allocator::operator=(Allocator&& o) noexcept {
    if (this != &o) {
        destroy();
        allocator_ = o.allocator_;
        device_ = o.device_;
        o.allocator_ = nullptr;
        o.device_ = nullptr;
    }
    return *this;
}

BudgetStats Allocator::device_local_budget() const {
    BudgetStats out;
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(device_->physical_handle(), &props);

    std::uint32_t heap = VK_MAX_MEMORY_HEAPS;
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            heap = props.memoryTypes[i].heapIndex;
            break;
        }
    }
    if (heap == VK_MAX_MEMORY_HEAPS) {
        return out;
    }

    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator_, budgets);
    out.budget_bytes = budgets[heap].budget;
    out.usage_bytes = budgets[heap].usage;
    out.allocation_bytes = budgets[heap].statistics.allocationBytes;
    out.block_bytes = budgets[heap].statistics.blockBytes;
    return out;
}

} // namespace vulcanGE::gpu
