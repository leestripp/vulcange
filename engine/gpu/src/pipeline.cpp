#include <vge/gpu/pipeline.hpp>

#include <array>
#include <fstream>

namespace vulcanGE::gpu {

std::expected<std::vector<std::uint32_t>, Error> load_spirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::unexpected(make_error(GpuError::shader_load_failed,
                                          VK_ERROR_INITIALIZATION_FAILED,
                                          "cannot open " + path.string()));
    }
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        return std::unexpected(make_error(GpuError::shader_load_failed,
                                          VK_ERROR_INITIALIZATION_FAILED,
                                          "bad SPIR-V size in " + path.string()));
    }
    file.seekg(0);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4);
    if (!file.read(reinterpret_cast<char*>(words.data()), size)) {
        return std::unexpected(make_error(GpuError::shader_load_failed,
                                          VK_ERROR_INITIALIZATION_FAILED,
                                          "cannot read " + path.string()));
    }
    return words;
}

// ---- ShaderModule ----

std::expected<ShaderModule, Error> ShaderModule::create(const Device& device,
                                                        std::span<const std::uint32_t> spirv,
                                                        std::string entry_point) {
    const VkShaderModuleCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    };
    ShaderModule out;
    if (const VkResult r = vkCreateShaderModule(device.handle(), &ci, nullptr, &out.module_);
        r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::shader_module_creation_failed, r,
                                          "vkCreateShaderModule failed"));
    }
    out.device_ = device.handle();
    out.entry_ = std::move(entry_point);
    return out;
}

void ShaderModule::destroy() noexcept {
    if (module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, module_, nullptr);
        module_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
    entry_.clear();
}

ShaderModule::~ShaderModule() { destroy(); }

ShaderModule::ShaderModule(ShaderModule&& o) noexcept
    : device_(o.device_), module_(o.module_), entry_(std::move(o.entry_)) {
    o.module_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
}

ShaderModule& ShaderModule::operator=(ShaderModule&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        module_ = o.module_;
        entry_ = std::move(o.entry_);
        o.module_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

// ---- GraphicsPipeline ----

std::expected<GraphicsPipeline, Error>
GraphicsPipeline::create(const Device& device, const GraphicsPipelineDesc& desc) {
    const bool mesh_path = desc.mesh != nullptr;
    if (mesh_path == (desc.vertex != nullptr)) {
        // Exactly one of the classic vertex path and the mesh shader path.
        return std::unexpected(make_error(
            GpuError::pipeline_creation_failed, VK_ERROR_INITIALIZATION_FAILED,
            "exactly one of vertex or mesh shader is required"));
    }
    if (desc.task != nullptr && !mesh_path) {
        return std::unexpected(make_error(GpuError::pipeline_creation_failed,
                                          VK_ERROR_INITIALIZATION_FAILED,
                                          "task shader requires the mesh shader path"));
    }

    const VkPipelineLayoutCreateInfo layout_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(desc.set_layouts.size()),
        .pSetLayouts = desc.set_layouts.data(),
        .pushConstantRangeCount = static_cast<std::uint32_t>(desc.push_constant_ranges.size()),
        .pPushConstantRanges = desc.push_constant_ranges.data(),
    };
    GraphicsPipeline out;
    out.device_ = device.handle();
    if (const VkResult r = vkCreatePipelineLayout(device.handle(), &layout_ci, nullptr,
                                                  &out.layout_);
        r != VK_SUCCESS) {
        return std::unexpected(make_error(GpuError::pipeline_layout_creation_failed, r,
                                          "vkCreatePipelineLayout failed"));
    }

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    if (desc.task != nullptr) {
        stages.push_back({.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                          .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
                          .module = desc.task->handle(),
                          .pName = desc.task->entry_point().c_str()});
    }
    stages.push_back({.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = mesh_path ? VkShaderStageFlagBits{VK_SHADER_STAGE_MESH_BIT_EXT}
                                         : VkShaderStageFlagBits{VK_SHADER_STAGE_VERTEX_BIT},
                      .module = mesh_path ? desc.mesh->handle() : desc.vertex->handle(),
                      .pName = mesh_path ? desc.mesh->entry_point().c_str()
                                         : desc.vertex->entry_point().c_str()});
    if (desc.fragment != nullptr) {
        stages.push_back({.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                          .module = desc.fragment->handle(),
                          .pName = desc.fragment->entry_point().c_str()});
    }

    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = desc.cull_mode,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0F,
    };
    const VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = static_cast<VkBool32>(desc.depth_test),
        .depthWriteEnable = static_cast<VkBool32>(desc.depth_write),
        .depthCompareOp = desc.depth_compare,
    };
    const VkPipelineColorBlendAttachmentState blend_attachment{
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(
        desc.color_formats.size(), blend_attachment);
    const VkPipelineColorBlendStateCreateInfo blending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<std::uint32_t>(blend_attachments.size()),
        .pAttachments = blend_attachments.data(),
    };
    const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    const VkPipelineRenderingCreateInfo rendering_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = static_cast<std::uint32_t>(desc.color_formats.size()),
        .pColorAttachmentFormats = desc.color_formats.data(),
        .depthAttachmentFormat = desc.depth_format,
    };
    const VkGraphicsPipelineCreateInfo pipeline_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering_ci,
        .stageCount = static_cast<std::uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &blending,
        .pDynamicState = &dynamic_state,
        .layout = out.layout_,
    };

    if (const VkResult r = vkCreateGraphicsPipelines(device.handle(), VK_NULL_HANDLE, 1,
                                                     &pipeline_ci, nullptr, &out.pipeline_);
        r != VK_SUCCESS) {
        out.destroy();
        return std::unexpected(make_error(GpuError::pipeline_creation_failed, r,
                                          "vkCreateGraphicsPipelines failed"));
    }
    if (!desc.debug_name.empty()) {
        device.set_object_name(VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<std::uint64_t>(out.pipeline_), desc.debug_name);
    }
    return out;
}

void GraphicsPipeline::destroy() noexcept {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

GraphicsPipeline::~GraphicsPipeline() { destroy(); }

GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& o) noexcept
    : device_(o.device_), pipeline_(o.pipeline_), layout_(o.layout_) {
    o.pipeline_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
}

GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        pipeline_ = o.pipeline_;
        layout_ = o.layout_;
        o.pipeline_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}


std::expected<ComputePipeline, Error> ComputePipeline::create(const Device& device,
                                                              const ComputePipelineDesc& desc) {
    if (desc.shader == nullptr) {
        return std::unexpected(make_error(GpuError::pipeline_creation_failed,
                                          VK_ERROR_INITIALIZATION_FAILED,
                                          "compute shader is required"));
    }

    const VkPipelineLayoutCreateInfo layout_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(desc.set_layouts.size()),
        .pSetLayouts = desc.set_layouts.data(),
        .pushConstantRangeCount =
            static_cast<std::uint32_t>(desc.push_constant_ranges.size()),
        .pPushConstantRanges = desc.push_constant_ranges.data(),
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (const VkResult r =
            vkCreatePipelineLayout(device.handle(), &layout_ci, nullptr, &layout);
        r != VK_SUCCESS) {
        return std::unexpected(
            make_error(GpuError::pipeline_creation_failed, r, "vkCreatePipelineLayout failed"));
    }

    const VkComputePipelineCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = desc.shader->handle(),
                  .pName = desc.shader->entry_point().c_str()},
        .layout = layout,
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (const VkResult r = vkCreateComputePipelines(device.handle(), VK_NULL_HANDLE, 1, &ci,
                                                    nullptr, &pipeline);
        r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device.handle(), layout, nullptr);
        return std::unexpected(
            make_error(GpuError::pipeline_creation_failed, r, "vkCreateComputePipelines failed"));
    }
    if (!desc.debug_name.empty()) {
        device.set_object_name(VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<std::uint64_t>(pipeline), desc.debug_name);
    }

    ComputePipeline out;
    out.device_ = device.handle();
    out.pipeline_ = pipeline;
    out.layout_ = layout;
    return out;
}

ComputePipeline::~ComputePipeline() { destroy(); }

ComputePipeline::ComputePipeline(ComputePipeline&& o) noexcept
    : device_(o.device_), pipeline_(o.pipeline_), layout_(o.layout_) {
    o.pipeline_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        pipeline_ = o.pipeline_;
        layout_ = o.layout_;
        o.pipeline_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE;
    }
    return *this;
}

void ComputePipeline::destroy() noexcept {
    if (device_ != VK_NULL_HANDLE) {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout_, nullptr);
        }
    }
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
}

} // namespace vulcanGE::gpu

