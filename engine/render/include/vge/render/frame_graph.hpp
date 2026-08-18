#pragma once

// vge::render::FrameGraph — Plan §5.1.
// Dynamic render graph owning transient resources (VMA-backed), automatic
// pipeline barriers via Synchronization2 (VkImageMemoryBarrier2), and
// vkCmdBeginRendering dynamic-rendering scopes per pass. Debug-utils labels
// annotate every pass (Plan §16.3).
//
// Current scope (Phase 2a): serial execution on the graphics queue, transient
// image allocation without aliasing, swapchain import + present transition.
// Roadmap: resource aliasing, pass culling, multi-queue scheduling (§5.5).

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <vge/gpu/allocator.hpp>
#include <vge/gpu/device.hpp>
#include <vge/gpu/error.hpp>
#include <vge/gpu/image.hpp>

namespace vulcanGE::render {

using TextureHandle = std::uint32_t;
inline constexpr TextureHandle invalid_texture = ~0u;

struct TextureDesc {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0; // augmented from pass usage at compile()
};

struct Attachment {
    TextureHandle texture = invalid_texture;
    VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp store_op = VK_ATTACHMENT_STORE_OP_STORE;
    VkClearValue clear{};
};

// Passed to pass execute callbacks during recording.
struct PassContext {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkExtent2D extent{}; // render area of the current pass

    [[nodiscard]] VkImageView texture_view(TextureHandle h) const;
    [[nodiscard]] VkImage texture_image(TextureHandle h) const;

    const class FrameGraph* graph = nullptr;
};

class FrameGraph {
public:
    FrameGraph() = default;

    // Transient texture: allocated at compile(), owned by the graph. The graph
    // must outlive GPU execution of the recorded frame (frame fence).
    [[nodiscard]] TextureHandle create_texture(std::string name, TextureDesc desc);

    // External texture (e.g. swapchain image). `initial_layout` is the layout
    // the image is in when the graph executes.
    [[nodiscard]] TextureHandle import_texture(std::string name, VkImage image,
                                               VkImageView view, VkFormat format,
                                               VkExtent2D extent, VkImageLayout initial_layout);

    using ExecuteFn = std::function<void(const PassContext&)>;

    // Graphics pass with dynamic-rendering attachments. `reads` are transitioned
    // to SHADER_READ_ONLY_OPTIMAL / SHADER_SAMPLED_READ before the pass.
    void add_graphics_pass(std::string name, std::span<const TextureHandle> reads,
                           std::span<const Attachment> color_targets,
                           std::optional<Attachment> depth_target, ExecuteFn fn);

    // Compute pass (§5.2 light cluster / particles). Reads and writes are
    // transitioned to GENERAL with SHADER_SAMPLED_READ / SHADER_STORAGE_WRITE.
    void add_compute_pass(std::string name, std::span<const TextureHandle> reads,
                          std::span<const TextureHandle> writes, ExecuteFn fn);

    // Marks an imported texture for presentation: after the last recorded pass
    // it transitions to PRESENT_SRC_KHR.
    void present(TextureHandle h);

    // Allocates transient resources; must be called before execute().
    [[nodiscard]] std::expected<void, gpu::Error> compile(const gpu::Allocator& allocator);

    // Records barriers + passes into cmd (already begun). Non-const: tracked
    // texture states (layout/stage/access) advance as barriers are emitted.
    void execute(VkCommandBuffer cmd);

    // Drops passes/resources and all transient allocations.
    void reset() noexcept;

    [[nodiscard]] VkImageView texture_view(TextureHandle h) const;
    [[nodiscard]] VkImage texture_image(TextureHandle h) const;
    [[nodiscard]] VkExtent2D texture_extent(TextureHandle h) const;

private:
    struct TextureEntry {
        std::string name;
        bool imported = false;
        TextureDesc desc;
        VkImageUsageFlags accumulated_usage = 0;
        // Resolved (transient: index into images_; imported: raw handles).
        std::uint32_t transient_index = ~0u;
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkExtent2D extent{};
        // Tracked state for barrier generation.
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = VK_ACCESS_2_NONE;
        bool presented = false;
    };

    struct Pass {
        std::string name;
        bool compute = false;
        std::vector<TextureHandle> reads;
        std::vector<Attachment> color_targets;
        std::optional<Attachment> depth_target;
        std::vector<TextureHandle> storage_writes;
        ExecuteFn fn;
    };

    std::vector<TextureEntry> textures_;
    std::vector<gpu::Image> transient_images_;
    std::vector<Pass> passes_;

    void transition(VkCommandBuffer cmd, TextureEntry& tex, VkImageLayout layout,
                    VkPipelineStageFlags2 stage, VkAccessFlags2 access) const;

    // Per-pass internal steps.
    void execute_graphics(VkCommandBuffer cmd, Pass& pass);
    void execute_compute(VkCommandBuffer cmd, Pass& pass);

    PFN_vkCmdBeginDebugUtilsLabelEXT begin_label_ = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT end_label_ = nullptr;
    bool debug_funcs_loaded_ = false;

    void load_debug_funcs(const gpu::Device& device);
};

} // namespace vulcanGE::render
