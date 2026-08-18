#pragma once

// vge::render::MeshletLodTree — meshlet LOD cluster DAG (Plan §7.2).
// Built from a single mesh: level 0 is the meshopt meshlet set; each level
// above partitions the previous level's clusters into groups of ~4 and
// simplifies each group (meshopt, locked borders, absolute error) into the
// parent level. Every cluster records the world-space error of the group it
// belongs to and of its parent group; the task shader renders the DAG cut
// {err(cluster) <= threshold_px < err(parent)} projected to screen space.
//
// All levels share the original vertex buffer (simplification only removes
// indices) and one concatenated GpuMeshlet/packed-index buffer set, so the
// whole DAG draws with the same shader path as a flat MeshletSet.

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include <vge/gpu/allocator.hpp>
#include <vge/gpu/error.hpp>
#include <vge/render/meshlet.hpp>

namespace vulcanGE::render {

class MeshletLodTree {
public:
    MeshletLodTree() = default;

    [[nodiscard]] static std::expected<MeshletLodTree, gpu::Error>
    create(const gpu::Allocator& allocator, std::span<const float> vertices,
           std::span<const std::uint32_t> indices, std::string debug_name);

    MeshletLodTree(const MeshletLodTree&) = delete;
    MeshletLodTree& operator=(const MeshletLodTree&) = delete;
    MeshletLodTree(MeshletLodTree&&) noexcept = default;
    MeshletLodTree& operator=(MeshletLodTree&&) noexcept = default;

    [[nodiscard]] VkBuffer meshlet_buffer() const { return meshlets_.handle(); }
    [[nodiscard]] VkBuffer vertex_buffer() const { return vertices_.handle(); }
    [[nodiscard]] VkBuffer vertex_indices_buffer() const { return vindices_.handle(); }
    [[nodiscard]] VkBuffer triangle_indices_buffer() const { return tindices_.handle(); }
    [[nodiscard]] VkBuffer lod_info_buffer() const { return lod_info_.handle(); }
    [[nodiscard]] std::uint32_t cluster_count() const { return cluster_count_; }
    [[nodiscard]] std::uint32_t vertex_count() const { return vertex_count_; }
    [[nodiscard]] std::uint32_t lod_levels() const { return lod_levels_; }

    // Per-cluster record in lod_info_buffer (4 uints, float-bits view):
    // {group_error, parent_error, lod_level, pad}.
    static constexpr std::uint32_t lod_info_uints = 4;

    // Host-side view of the same records (buffer is persistently mapped).
    [[nodiscard]] const std::uint32_t* lod_info_data() const {
        return reinterpret_cast<const std::uint32_t*>(lod_info_.mapped());
    }

private:
    gpu::Buffer meshlets_;
    gpu::Buffer vertices_;
    gpu::Buffer vindices_;
    gpu::Buffer tindices_;
    gpu::Buffer lod_info_;
    std::uint32_t cluster_count_ = 0;
    std::uint32_t vertex_count_ = 0;
    std::uint32_t lod_levels_ = 0;
};

} // namespace vulcanGE::render
