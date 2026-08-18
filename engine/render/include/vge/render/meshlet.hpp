#pragma once

// vge::render::MeshletSet — meshlet cluster build via meshoptimizer (Plan
// §7.2). Splits an indexed triangle mesh into meshlets (<=64 verts / <=124
// tris) with per-cluster culling bounds (bounding sphere + backface cone) for
// the task-shader cluster culling path (Plan §5.3).
//
// LOD roadmap: single LOD level for now; the cluster DAG + screen-space error
// selection (§7.1/§7.2) lands with the LOD phase.

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <vge/gpu/allocator.hpp>
#include <vge/gpu/buffer.hpp>
#include <vge/gpu/error.hpp>

namespace vulcanGE::render {

// GPU meshlet descriptor (std430, 64 bytes). Matches Meshlet in meshlet.task.
struct GpuMeshlet {
    std::uint32_t vertex_offset;   // first entry in meshlet_vertices
    std::uint32_t triangle_offset; // first entry in meshlet_triangles
    std::uint32_t vertex_count;
    std::uint32_t triangle_count;
    float center[3];     // bounding sphere
    float radius;
    float cone_apex[3];  // backface cone (meshopt convention)
    float cone_cutoff;
    float cone_axis[3];
    float pad;
};
static_assert(sizeof(GpuMeshlet) == 64);

// Vertex layout shared with mesh.vert / meshlet.mesh: 8 floats
// (px py pz nx ny nz u v).
inline constexpr std::uint32_t vertex_floats = 8;

class MeshletSet {
public:
    MeshletSet() = default;

    // Builds meshlets from an indexed triangle mesh. `vertices` is the full
    // interleaved vertex data (vertex_floats per vertex); positions must be
    // the first 3 floats. `indices` is the uint32 triangle index buffer.
    [[nodiscard]] static std::expected<MeshletSet, gpu::Error>
    create(const gpu::Allocator& allocator, std::span<const float> vertices,
           std::span<const std::uint32_t> indices, std::string debug_name = {});

    [[nodiscard]] VkBuffer vertex_buffer() const { return vertices_.handle(); }
    [[nodiscard]] VkBuffer meshlet_buffer() const { return meshlets_.handle(); }
    [[nodiscard]] VkBuffer vertex_indices_buffer() const { return vertex_indices_.handle(); }
    [[nodiscard]] VkBuffer triangle_indices_buffer() const { return triangle_indices_.handle(); }
    [[nodiscard]] std::uint32_t meshlet_count() const noexcept { return meshlet_count_; }
    [[nodiscard]] std::uint32_t vertex_count() const noexcept { return vertex_count_; }

private:
    gpu::Buffer vertices_;
    gpu::Buffer meshlets_;
    gpu::Buffer vertex_indices_;   // global vertex indices per meshlet
    gpu::Buffer triangle_indices_; // packed uint32 per triangle (3x uint8 local)
    std::uint32_t meshlet_count_ = 0;
    std::uint32_t vertex_count_ = 0;
};

} // namespace vulcanGE::render
