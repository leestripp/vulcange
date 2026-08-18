#include <vge/render/frame_graph.hpp>

#include <array>

namespace vulcanGE::render {

namespace {

// State a texture must be in for a given pass usage (Plan §5.1 auto-barriers).
struct RequiredState {
    VkImageLayout layout;
    VkPipelineStageFlags2 stage;
    VkAccessFlags2 access;
    VkImageUsageFlags usage_bit;
};

[[nodiscard]] RequiredState state_for_read() {
    return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_USAGE_SAMPLED_BIT};
}
[[nodiscard]] RequiredState state_for_color_write(VkAttachmentLoadOp load) {
    VkAccessFlags2 access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (load == VK_ATTACHMENT_LOAD_OP_LOAD) {
        access |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    }
    return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, access,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
}
[[nodiscard]] RequiredState state_for_depth_write() {
    return {VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT};
}
[[nodiscard]] RequiredState state_for_storage_write() {
    return {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_USAGE_STORAGE_BIT};
}

} // namespace

VkImageView PassContext::texture_view(TextureHandle h) const { return graph->texture_view(h); }
VkImage PassContext::texture_image(TextureHandle h) const { return graph->texture_image(h); }

TextureHandle FrameGraph::create_texture(std::string name, TextureDesc desc) {
    TextureEntry e;
    e.name = std::move(name);
    e.imported = false;
    e.desc = desc;
    e.extent = {desc.width, desc.height};
    textures_.push_back(std::move(e));
    return static_cast<TextureHandle>(textures_.size() - 1);
}

TextureHandle FrameGraph::import_texture(std::string name, VkImage image, VkImageView view,
                                         VkFormat format, VkExtent2D extent,
                                         VkImageLayout initial_layout) {
    TextureEntry e;
    e.name = std::move(name);
    e.imported = true;
    e.desc = {extent.width, extent.height, format, 0};
    e.image = image;
    e.view = view;
    e.extent = extent;
    e.layout = initial_layout;
    textures_.push_back(std::move(e));
    return static_cast<TextureHandle>(textures_.size() - 1);
}

void FrameGraph::add_graphics_pass(std::string name, std::span<const TextureHandle> reads,
                                   std::span<const Attachment> color_targets,
                                   std::optional<Attachment> depth_target, ExecuteFn fn) {
    Pass p;
    p.name = std::move(name);
    p.compute = false;
    p.reads.assign(reads.begin(), reads.end());
    p.color_targets.assign(color_targets.begin(), color_targets.end());
    p.depth_target = depth_target;
    p.fn = std::move(fn);
    for (const TextureHandle h : p.reads) {
        textures_.at(h).accumulated_usage |= state_for_read().usage_bit;
    }
    for (const Attachment& a : p.color_targets) {
        textures_.at(a.texture).accumulated_usage |= state_for_color_write(a.load_op).usage_bit;
    }
    if (p.depth_target) {
        textures_.at(p.depth_target->texture).accumulated_usage |= state_for_depth_write().usage_bit;
    }
    passes_.push_back(std::move(p));
}

void FrameGraph::add_compute_pass(std::string name, std::span<const TextureHandle> reads,
                                  std::span<const TextureHandle> writes, ExecuteFn fn) {
    Pass p;
    p.name = std::move(name);
    p.compute = true;
    p.reads.assign(reads.begin(), reads.end());
    p.storage_writes.assign(writes.begin(), writes.end());
    p.fn = std::move(fn);
    for (const TextureHandle h : p.reads) {
        textures_.at(h).accumulated_usage |= state_for_read().usage_bit;
    }
    for (const TextureHandle h : p.storage_writes) {
        textures_.at(h).accumulated_usage |= state_for_storage_write().usage_bit;
    }
    passes_.push_back(std::move(p));
}

void FrameGraph::present(TextureHandle h) { textures_.at(h).presented = true; }

std::expected<void, gpu::Error> FrameGraph::compile(const gpu::Allocator& allocator) {
    for (TextureEntry& e : textures_) {
        if (e.imported) {
            continue;
        }
        if (e.accumulated_usage == 0) {
            continue; // declared but unused — pass culling later (§5.1 roadmap)
        }
        auto img = gpu::Image::create(allocator,
                                      {.width = e.desc.width,
                                       .height = e.desc.height,
                                       .format = e.desc.format,
                                       .usage = e.desc.usage | e.accumulated_usage,
                                       .debug_name = "fg_" + e.name});
        if (!img) {
            return std::unexpected(img.error());
        }
        e.transient_index = static_cast<std::uint32_t>(transient_images_.size());
        e.image = img->handle();
        e.view = img->view();
        transient_images_.push_back(std::move(*img));
    }
    load_debug_funcs(allocator.device());
    return {};
}

void FrameGraph::transition(VkCommandBuffer cmd, TextureEntry& tex, VkImageLayout layout,
                            VkPipelineStageFlags2 stage, VkAccessFlags2 access) const {
    if (tex.layout == layout && tex.access == access && tex.stage == stage) {
        return;
    }
    const VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = tex.stage,
        .srcAccessMask = tex.access,
        .dstStageMask = stage,
        .dstAccessMask = access,
        .oldLayout = tex.layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = tex.image,
        .subresourceRange = {gpu::Image::is_depth_format(tex.desc.format)
                                 ? VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT}
                                 : VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT},
                             0, 1, 0, 1},
    };
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
    tex.layout = layout;
    tex.stage = stage;
    tex.access = access;
}

void FrameGraph::execute_graphics(VkCommandBuffer cmd, Pass& pass) {
    std::vector<VkRenderingAttachmentInfo> color_infos;
    color_infos.reserve(pass.color_targets.size());
    VkExtent2D extent{};

    for (const TextureHandle h : pass.reads) {
        TextureEntry& e = textures_.at(h);
        const RequiredState s = state_for_read();
        transition(cmd, e, s.layout, s.stage, s.access);
    }
    for (const Attachment& a : pass.color_targets) {
        TextureEntry& e = textures_.at(a.texture);
        const RequiredState s = state_for_color_write(a.load_op);
        transition(cmd, e, s.layout, s.stage, s.access);
        extent = e.extent;
        color_infos.push_back({.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                               .imageView = e.view,
                               .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               .loadOp = a.load_op,
                               .storeOp = a.store_op,
                               .clearValue = a.clear});
    }

    VkRenderingAttachmentInfo depth_info{};
    if (pass.depth_target) {
        const Attachment& a = *pass.depth_target;
        TextureEntry& e = textures_.at(a.texture);
        const RequiredState s = state_for_depth_write();
        transition(cmd, e, s.layout, s.stage, s.access);
        extent = e.extent;
        depth_info = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                      .imageView = e.view,
                      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                      .loadOp = a.load_op,
                      .storeOp = a.store_op,
                      .clearValue = a.clear};
    }

    const VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<std::uint32_t>(color_infos.size()),
        .pColorAttachments = color_infos.data(),
        .pDepthAttachment = pass.depth_target ? &depth_info : nullptr,
    };
    vkCmdBeginRendering(cmd, &ri);

    const PassContext ctx{.cmd = cmd, .extent = extent, .graph = this};
    pass.fn(ctx);

    vkCmdEndRendering(cmd);
}

void FrameGraph::execute_compute(VkCommandBuffer cmd, Pass& pass) {
    for (const TextureHandle h : pass.reads) {
        TextureEntry& e = textures_.at(h);
        const RequiredState s = state_for_read();
        transition(cmd, e, s.layout, s.stage, s.access);
    }
    for (const TextureHandle h : pass.storage_writes) {
        TextureEntry& e = textures_.at(h);
        const RequiredState s = state_for_storage_write();
        transition(cmd, e, s.layout, s.stage, s.access);
    }
    const VkExtent2D extent =
        !pass.storage_writes.empty() ? textures_.at(pass.storage_writes.front()).extent
                                     : VkExtent2D{};
    const PassContext ctx{.cmd = cmd, .extent = extent, .graph = this};
    pass.fn(ctx);
}

void FrameGraph::execute(VkCommandBuffer cmd) {
    for (Pass& pass : passes_) {
        if (begin_label_ != nullptr) {
            const VkDebugUtilsLabelEXT label{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                .pLabelName = pass.name.c_str(),
            };
            begin_label_(cmd, &label);
        }
        if (pass.compute) {
            execute_compute(cmd, pass);
        } else {
            execute_graphics(cmd, pass);
        }
        if (end_label_ != nullptr) {
            end_label_(cmd);
        }
    }

    // Final present transitions (Plan §5.4: swapchain -> PRESENT_SRC_KHR).
    for (TextureEntry& e : textures_) {
        if (e.presented) {
            transition(cmd, e, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_NONE,
                       VK_ACCESS_2_NONE);
        }
    }
}

void FrameGraph::reset() noexcept {
    textures_.clear();
    passes_.clear();
    transient_images_.clear();
}

VkImageView FrameGraph::texture_view(TextureHandle h) const { return textures_.at(h).view; }
VkImage FrameGraph::texture_image(TextureHandle h) const { return textures_.at(h).image; }
VkExtent2D FrameGraph::texture_extent(TextureHandle h) const { return textures_.at(h).extent; }

void FrameGraph::load_debug_funcs(const gpu::Device& device) {
    if (debug_funcs_loaded_) {
        return;
    }
    begin_label_ = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(device.handle(), "vkCmdBeginDebugUtilsLabelEXT"));
    end_label_ = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(device.handle(), "vkCmdEndDebugUtilsLabelEXT"));
    debug_funcs_loaded_ = true;
}

} // namespace vulcanGE::render
