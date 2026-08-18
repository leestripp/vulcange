#include <vge/gpu/device.hpp>

#include <algorithm>
#include <cstring>
#include <optional>
#include <set>

namespace vulcanGE::gpu {

namespace {

[[nodiscard]] bool has_extension(VkPhysicalDevice pd, std::string_view name) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, exts.data());
    for (const auto& e : exts) {
        if (name == e.extensionName) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::uint64_t dedicated_vram(VkPhysicalDevice pd) {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            total += mem.memoryHeaps[i].size;
        }
    }
    return total;
}

[[nodiscard]] std::expected<QueueFamilies, GpuError>
select_queues(VkPhysicalDevice pd, VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());

    QueueFamilies q;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& f = families[i];
        const bool graphics = (f.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool present = true;
        if (surface != VK_NULL_HANDLE) {
            VkBool32 sup = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &sup);
            present = sup == VK_TRUE;
        }
        if (graphics && present && q.graphics == ~0u) {
            q.graphics = i;
        }
        // Dedicated compute: compute bit WITHOUT graphics bit.
        if ((f.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && !graphics && q.compute == ~0u) {
            q.compute = i;
            q.dedicated_compute = true;
        }
        // Dedicated transfer: transfer bit WITHOUT graphics/compute bits.
        if ((f.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 &&
            (f.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0 &&
            q.transfer == ~0u) {
            q.transfer = i;
            q.dedicated_transfer = true;
        }
    }
    if (q.graphics == ~0u) {
        // Fall back: graphics-only family without present (headless), or fail.
        for (std::uint32_t i = 0; i < count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                q.graphics = i;
                break;
            }
        }
        if (q.graphics == ~0u) {
            return std::unexpected(GpuError::no_graphics_queue_family);
        }
    }
    if (q.compute == ~0u) {
        q.compute = q.graphics; // shared fallback (Plan §5.5)
    }
    if (q.transfer == ~0u) {
        q.transfer = q.graphics;
    }
    return q;
}

struct Candidate {
    VkPhysicalDevice pd;
    GpuInfo info;
    int score;
};

[[nodiscard]] std::optional<Candidate>
evaluate(VkPhysicalDevice pd, VkSurfaceKHR surface, const DeviceDesc& desc) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    if (props.apiVersion < VK_API_VERSION_1_3) {
        return std::nullopt; // Plan §5.6 hard floor
    }

    auto queues = select_queues(pd, surface);
    if (!queues) {
        return std::nullopt;
    }

    if (surface != VK_NULL_HANDLE && !has_extension(pd, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        return std::nullopt;
    }

    GpuInfo info;
    info.name = props.deviceName;
    info.type = props.deviceType;
    info.api_version = props.apiVersion;
    info.limits = props.limits;
    info.dedicated_vram_bytes = dedicated_vram(pd);
    info.queues = *queues;
    info.features = {
        .mesh_shader = has_extension(pd, VK_EXT_MESH_SHADER_EXTENSION_NAME),
        .descriptor_buffer = has_extension(pd, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME),
        // Core since Vulkan 1.2 (min spec is 1.3); extension string may be absent.
        .descriptor_indexing = info.api_version >= VK_API_VERSION_1_2 ||
                               has_extension(pd, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME),
        .ray_tracing_pipeline = has_extension(pd, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME),
        .present_id = has_extension(pd, VK_KHR_PRESENT_ID_EXTENSION_NAME),
        .present_wait = has_extension(pd, VK_KHR_PRESENT_WAIT_EXTENSION_NAME),
        .memory_budget = has_extension(pd, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME),
    };

    if (info.dedicated_vram_bytes < desc.min_dedicated_vram && !desc.allow_below_min_spec) {
        return std::nullopt;
    }

    // Scoring: discrete strongly preferred; VRAM breaks ties.
    int score = 0;
    switch (props.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 1000; break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 100; break;
    default: score = 10; break;
    }
    score += static_cast<int>(std::min<std::uint64_t>(info.dedicated_vram_bytes >> 30, 100));
    if (queues->dedicated_compute) score += 10;
    if (queues->dedicated_transfer) score += 10;
    if (info.features.mesh_shader) score += 5;
    if (info.features.descriptor_buffer) score += 5;

    return Candidate{pd, std::move(info), score};
}

} // namespace

std::expected<Device, Error> Device::create(const Instance& instance, const DeviceDesc& desc) {
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance.handle(), &count, nullptr);
    if (count == 0) {
        return std::unexpected(make_error(GpuError::no_suitable_physical_device, VK_ERROR_INITIALIZATION_FAILED));
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance.handle(), &count, devices.data());

    std::optional<Candidate> best;
    for (VkPhysicalDevice pd : devices) {
        auto cand = evaluate(pd, desc.surface, desc);
        if (cand && (!best || cand->score > best->score)) {
            best = std::move(*cand);
        }
    }
    if (!best) {
        return std::unexpected(
            make_error(GpuError::no_suitable_physical_device, VK_ERROR_INITIALIZATION_FAILED,
                       "no device met Vulkan 1.3 + 8 GiB VRAM min spec (Plan §5.6)"));
    }

    // One queue per distinct family (Plan §5.5).
    const auto& q = best->info.queues;
    const std::set<std::uint32_t> unique_families{q.graphics, q.compute, q.transfer};
    const float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queue_cis;
    for (const std::uint32_t fam : unique_families) {
        queue_cis.push_back(VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = fam,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        });
    }

    // Core 1.2 features for the bindless descriptor architecture (Plan §5.3):
    // global texture arrays with update-after-bind + non-uniform indexing.
    VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
        .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
        .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
        .descriptorBindingPartiallyBound = VK_TRUE,
        .descriptorBindingVariableDescriptorCount = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE,
        .scalarBlockLayout = VK_TRUE,
    };
    // Core 1.3 features the renderer relies on (dynamic rendering + sync2).
    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = nullptr,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };
    features12.pNext = &features13;
    VkPhysicalDeviceFeatures2 features2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features12,
    };

    std::vector<const char*> extensions;
    if (desc.surface != VK_NULL_HANDLE) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkDeviceCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features2,
        .queueCreateInfoCount = static_cast<std::uint32_t>(queue_cis.size()),
        .pQueueCreateInfos = queue_cis.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    Device out;
    if (const VkResult r = vkCreateDevice(best->pd, &ci, nullptr, &out.device_); r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::device_creation_failed, r));
    }
    out.physical_ = best->pd;
    out.info_ = std::move(best->info);
    vkGetDeviceQueue(out.device_, q.graphics, 0, &out.graphics_queue_);
    vkGetDeviceQueue(out.device_, q.compute, 0, &out.compute_queue_);
    vkGetDeviceQueue(out.device_, q.transfer, 0, &out.transfer_queue_);
    return out;
}

Device::~Device() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
}

Device::Device(Device&& o) noexcept
    : device_(o.device_), physical_(o.physical_),
      graphics_queue_(o.graphics_queue_), compute_queue_(o.compute_queue_),
      transfer_queue_(o.transfer_queue_), info_(std::move(o.info_)) {
    o.device_ = VK_NULL_HANDLE;
    o.physical_ = VK_NULL_HANDLE;
    o.graphics_queue_ = o.compute_queue_ = o.transfer_queue_ = VK_NULL_HANDLE;
}

Device& Device::operator=(Device&& o) noexcept {
    if (this != &o) {
        this->~Device();
        device_ = o.device_;
        physical_ = o.physical_;
        graphics_queue_ = o.graphics_queue_;
        compute_queue_ = o.compute_queue_;
        transfer_queue_ = o.transfer_queue_;
        info_ = std::move(o.info_);
        o.device_ = VK_NULL_HANDLE;
        o.physical_ = VK_NULL_HANDLE;
        o.graphics_queue_ = o.compute_queue_ = o.transfer_queue_ = VK_NULL_HANDLE;
    }
    return *this;
}

void Device::set_object_name(VkObjectType type, std::uint64_t object, std::string_view name) const {
    if (device_ == VK_NULL_HANDLE) {
        return;
    }
    auto set_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));
    if (set_name == nullptr) {
        return;
    }
    const std::string name_str{name};
    const VkDebugUtilsObjectNameInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .objectType = type,
        .objectHandle = object,
        .pObjectName = name_str.c_str(),
    };
    set_name(device_, &info);
}

} // namespace vulcanGE::gpu
