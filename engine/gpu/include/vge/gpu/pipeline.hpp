#pragma once

// vge::gpu shader module + graphics pipeline for dynamic rendering (no
// VkRenderPass/VkFramebuffer — Plan §5.1). Fixed function state is baked with
// sensible defaults: dynamic viewport/scissor, no vertex input (vertex pulling
// via storage buffers is the §5.3 path), back-face culling, reverse-Z depth
// (GREATER, Plan §3.3).

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <vge/gpu/device.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::gpu {

// Reads a .spv file into a 4-byte-aligned word vector.
[[nodiscard]] std::expected<std::vector<std::uint32_t>, Error>
load_spirv(const std::filesystem::path& path);

class ShaderModule {
public:
    ShaderModule() = default;
    ~ShaderModule();
    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&& o) noexcept;
    ShaderModule& operator=(ShaderModule&& o) noexcept;

    [[nodiscard]] static std::expected<ShaderModule, Error>
    create(const Device& device, std::span<const std::uint32_t> spirv,
           std::string entry_point = "main");

    [[nodiscard]] VkShaderModule handle() const noexcept { return module_; }
    [[nodiscard]] const std::string& entry_point() const noexcept { return entry_; }

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkShaderModule module_ = VK_NULL_HANDLE;
    std::string entry_;

    void destroy() noexcept;
};

struct GraphicsPipelineDesc {
    // Classic path: vertex is required (mesh/task must be null).
    // Mesh shader path (Plan §5.3 primary): mesh required, task optional
    // (cluster culling), vertex must be null.
    const ShaderModule* vertex = nullptr;
    const ShaderModule* task = nullptr; // VK_EXT_mesh_shader task (amplification)
    const ShaderModule* mesh = nullptr; // VK_EXT_mesh_shader mesh
    const ShaderModule* fragment = nullptr; // null = depth-only pass
    std::vector<VkDescriptorSetLayout> set_layouts;
    std::vector<VkPushConstantRange> push_constant_ranges;
    std::vector<VkFormat> color_formats;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    bool depth_test = false;
    bool depth_write = false;
    VkCompareOp depth_compare = VK_COMPARE_OP_GREATER; // reverse-Z (§3.3)
    VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT;
    // Standard alpha blending (src.a * src + (1 - src.a) * dst) on every
    // color attachment — used by the deferred decal pass (§6.5).
    bool blend = false;
    std::string debug_name;
};

struct ComputePipelineDesc {
    const ShaderModule* shader = nullptr; // required
    std::vector<VkDescriptorSetLayout> set_layouts;
    std::vector<VkPushConstantRange> push_constant_ranges;
    std::string debug_name;
};

class ComputePipeline {
public:
    ComputePipeline() = default;
    ~ComputePipeline();
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&& o) noexcept;
    ComputePipeline& operator=(ComputePipeline&& o) noexcept;

    [[nodiscard]] static std::expected<ComputePipeline, Error>
    create(const Device& device, const ComputePipelineDesc& desc);

    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    void destroy() noexcept;
};

class GraphicsPipeline {
public:
    GraphicsPipeline() = default;
    ~GraphicsPipeline();
    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(GraphicsPipeline&& o) noexcept;
    GraphicsPipeline& operator=(GraphicsPipeline&& o) noexcept;

    [[nodiscard]] static std::expected<GraphicsPipeline, Error>
    create(const Device& device, const GraphicsPipelineDesc& desc);

    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE; // borrowed from Device (outlives us)
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    void destroy() noexcept;
};

} // namespace vulcanGE::gpu
