#include <vge/gpu/instance.hpp>

#include <cstdio>
#include <vector>

namespace vulcanGE::gpu {

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_messenger_trampoline(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* user_data) {
    const auto* cb = static_cast<const DebugCallback*>(user_data);
    if (*cb) {
        (*cb)(severity, data != nullptr ? std::string_view{data->pMessage} : std::string_view{});
    }
    return VK_FALSE;
}

[[nodiscard]] bool layer_available(std::string_view name) {
    std::uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers) {
        if (name == l.layerName) {
            return true;
        }
    }
    return false;
}

} // namespace

std::string_view to_string(GpuError e) noexcept {
    switch (e) {
    case GpuError::instance_creation_failed: return "VkInstance creation failed";
    case GpuError::validation_layer_unavailable: return "VK_LAYER_KHRONOS_validation unavailable";
    case GpuError::debug_utils_unavailable: return "VK_EXT_debug_utils unavailable";
    case GpuError::no_vulkan_driver: return "no Vulkan driver / ICD found";
    case GpuError::no_suitable_physical_device: return "no suitable physical device";
    case GpuError::api_version_too_low: return "device does not support Vulkan 1.3 (Plan §5.6 min spec)";
    case GpuError::vram_below_min_spec: return "dedicated VRAM below 8 GiB min spec (Plan §5.6)";
    case GpuError::no_graphics_queue_family: return "no graphics queue family";
    case GpuError::device_creation_failed: return "VkDevice creation failed";
    case GpuError::swapchain_support_missing: return "VK_KHR_swapchain not supported";
    case GpuError::swapchain_creation_failed: return "VkSwapchainKHR creation failed";
    case GpuError::no_present_mode_available: return "no usable present mode";
    case GpuError::no_surface_format_available: return "no usable surface format";
    case GpuError::allocator_creation_failed: return "VmaAllocator creation failed";
    case GpuError::buffer_creation_failed: return "VkBuffer/VMA allocation failed";
    case GpuError::image_creation_failed: return "VkImage/VMA allocation failed";
    case GpuError::image_view_creation_failed: return "VkImageView creation failed";
    case GpuError::command_pool_creation_failed: return "VkCommandPool creation failed";
    case GpuError::command_buffer_allocation_failed: return "VkCommandBuffer allocation failed";
    case GpuError::sync_object_creation_failed: return "fence/semaphore creation failed";
    case GpuError::shader_load_failed: return "SPIR-V file load failed";
    case GpuError::shader_module_creation_failed: return "VkShaderModule creation failed";
    case GpuError::pipeline_layout_creation_failed: return "VkPipelineLayout creation failed";
    case GpuError::pipeline_creation_failed: return "VkPipeline creation failed";
    }
    return "unknown GPU error";
}

std::string Error::message() const {
    std::string msg{to_string(code)};
    if (vk_result != VK_SUCCESS) {
        msg += " (VkResult " + std::to_string(static_cast<int>(vk_result)) + ")";
    }
    if (!detail.empty()) {
        msg += ": " + detail;
    }
    return msg;
}

std::expected<Instance, Error> Instance::create(const InstanceDesc& desc) {
    std::uint32_t api = 0;
    if (const VkResult r = vkEnumerateInstanceVersion(&api); r != VK_SUCCESS || api < VK_API_VERSION_1_3) {
        return std::unexpected(
            make_error(GpuError::no_vulkan_driver, r, "instance-level Vulkan 1.3 required"));
    }

    const bool want_validation = desc.enable_validation;
    const bool have_validation =
        want_validation && layer_available("VK_LAYER_KHRONOS_validation");
    if (want_validation && !have_validation) {
        // Warn but continue — validation must never block a dev machine boot.
        std::fprintf(stderr, "[vge::gpu] warning: %s\n", to_string(GpuError::validation_layer_unavailable).data());
    }

    std::vector<const char*> extensions(desc.required_extensions.begin(),
                                        desc.required_extensions.end());
    if (have_validation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const std::string app_name{desc.app_name};
    VkApplicationInfo app{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = app_name.c_str(),
        .applicationVersion = desc.app_version,
        .pEngineName = "vulcanGE",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    // Static storage: callback must outlive the instance.
    static DebugCallback callback_storage;
    callback_storage = desc.debug_callback
        ? desc.debug_callback
        : DebugCallback{[](VkDebugUtilsMessageSeverityFlagBitsEXT sev, std::string_view msg) {
              const char* tag = sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR"
                                : sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "WARN"
                                                                                         : "info";
              std::fprintf(stderr, "[vulkan %s] %.*s\n", tag, static_cast<int>(msg.size()), msg.data());
          }};

    VkDebugUtilsMessengerCreateInfoEXT messenger_ci{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_messenger_trampoline,
        .pUserData = &callback_storage,
    };

    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = have_validation ? &messenger_ci : nullptr, // catch instance-level issues
        .pApplicationInfo = &app,
        .enabledLayerCount = have_validation ? 1u : 0u,
        .ppEnabledLayerNames = have_validation ? layers : nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    Instance out;
    if (const VkResult r = vkCreateInstance(&ci, nullptr, &out.instance_); r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::instance_creation_failed, r));
    }
    out.validation_ = have_validation;

    if (have_validation) {
        auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(out.instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create_messenger == nullptr ||
            create_messenger(out.instance_, &messenger_ci, nullptr, &out.messenger_) != VK_SUCCESS) {
            return std::unexpected(make_error(GpuError::debug_utils_unavailable, VK_ERROR_EXTENSION_NOT_PRESENT));
        }
    }
    return out;
}

Instance::~Instance() {
    if (instance_ != VK_NULL_HANDLE) {
        if (messenger_ != VK_NULL_HANDLE) {
            auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy != nullptr) {
                destroy(instance_, messenger_, nullptr);
            }
        }
        vkDestroyInstance(instance_, nullptr);
    }
}

Instance::Instance(Instance&& o) noexcept
    : instance_(o.instance_), messenger_(o.messenger_), validation_(o.validation_) {
    o.instance_ = VK_NULL_HANDLE;
    o.messenger_ = VK_NULL_HANDLE;
    o.validation_ = false;
}

Instance& Instance::operator=(Instance&& o) noexcept {
    if (this != &o) {
        this->~Instance();
        instance_ = o.instance_;
        messenger_ = o.messenger_;
        validation_ = o.validation_;
        o.instance_ = VK_NULL_HANDLE;
        o.messenger_ = VK_NULL_HANDLE;
        o.validation_ = false;
    }
    return *this;
}

} // namespace vulcanGE::gpu
