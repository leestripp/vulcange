#include <vge/render/bindless_table.hpp>

#include <array>

namespace vulcanGE::render {

std::expected<BindlessTable, gpu::Error>
BindlessTable::create(const gpu::Device& device, std::uint32_t max_textures) {
    const std::array bindings{
        VkDescriptorSetLayoutBinding{
            .binding = frame_ubo_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                          VK_SHADER_STAGE_COMPUTE_BIT,
        },
        VkDescriptorSetLayoutBinding{
            .binding = storage_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                          VK_SHADER_STAGE_COMPUTE_BIT,
        },
        VkDescriptorSetLayoutBinding{
            .binding = textures_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = max_textures,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                          VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    // Binding flags must cover every binding (indexed by binding position).
    const std::array binding_flags{
        VkDescriptorBindingFlags{0},
        VkDescriptorBindingFlags{0},
        VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                 VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                 VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT},
    };
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(binding_flags.size()),
        .pBindingFlags = binding_flags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo layout_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flags_ci,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    BindlessTable out;
    out.device_ = device.handle();
    out.max_textures_ = max_textures;
    if (const VkResult r =
            vkCreateDescriptorSetLayout(device.handle(), &layout_ci, nullptr, &out.layout_);
        r != VK_SUCCESS) {
        return std::unexpected(gpu::make_error(gpu::GpuError::pipeline_layout_creation_failed, r,
                                               "vkCreateDescriptorSetLayout failed"));
    }

    const std::array pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_textures},
    };
    const VkDescriptorPoolCreateInfo pool_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };
    if (const VkResult r =
            vkCreateDescriptorPool(device.handle(), &pool_ci, nullptr, &out.pool_);
        r != VK_SUCCESS) {
        out.destroy();
        return std::unexpected(gpu::make_error(gpu::GpuError::pipeline_creation_failed, r,
                                               "vkCreateDescriptorPool failed"));
    }

    const VkDescriptorSetVariableDescriptorCountAllocateInfo variable_count{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
        .descriptorSetCount = 1,
        .pDescriptorCounts = &max_textures,
    };
    const VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = &variable_count,
        .descriptorPool = out.pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &out.layout_,
    };
    if (const VkResult r = vkAllocateDescriptorSets(device.handle(), &alloc_info, &out.set_);
        r != VK_SUCCESS) {
        out.destroy();
        return std::unexpected(gpu::make_error(gpu::GpuError::pipeline_creation_failed, r,
                                               "vkAllocateDescriptorSets failed"));
    }
    return out;
}

void BindlessTable::bind_frame_ubo(VkBuffer buffer, VkDeviceSize size,
                                   VkDeviceSize offset) const {
    const VkDescriptorBufferInfo info{.buffer = buffer, .offset = offset, .range = size};
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set_,
        .dstBinding = frame_ubo_binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &info,
    };
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void BindlessTable::bind_storage_buffer(VkBuffer buffer, VkDeviceSize size,
                                        VkDeviceSize offset) const {
    const VkDescriptorBufferInfo info{.buffer = buffer, .offset = offset, .range = size};
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set_,
        .dstBinding = storage_binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &info,
    };
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

std::uint32_t BindlessTable::register_texture(VkImageView view, VkSampler sampler) {
    const std::uint32_t index = next_texture_++;
    const VkDescriptorImageInfo info{
        .sampler = sampler,
        .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set_,
        .dstBinding = textures_binding,
        .dstArrayElement = index,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &info,
    };
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    return index;
}

void BindlessTable::destroy() noexcept {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr); // frees the set
        pool_ = VK_NULL_HANDLE;
        set_ = VK_NULL_HANDLE;
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

BindlessTable::~BindlessTable() { destroy(); }

BindlessTable::BindlessTable(BindlessTable&& o) noexcept
    : device_(o.device_), pool_(o.pool_), layout_(o.layout_), set_(o.set_),
      max_textures_(o.max_textures_), next_texture_(o.next_texture_) {
    o.pool_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.set_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
}

BindlessTable& BindlessTable::operator=(BindlessTable&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        pool_ = o.pool_;
        layout_ = o.layout_;
        set_ = o.set_;
        max_textures_ = o.max_textures_;
        next_texture_ = o.next_texture_;
        o.pool_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE;
        o.set_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vulcanGE::render
