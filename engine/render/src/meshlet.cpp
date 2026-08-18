#include <vge/render/meshlet.hpp>

#include <cstring>

// meshoptimizer.h is third-party; relax warnings for this include only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <meshoptimizer.h>
#pragma GCC diagnostic pop

namespace vulcanGE::render {

namespace {
constexpr std::size_t max_meshlet_vertices = 64;
constexpr std::size_t max_meshlet_triangles = 124;
} // namespace

std::expected<MeshletSet, gpu::Error>
MeshletSet::create(const gpu::Allocator& allocator, std::span<const float> vertices,
                   std::span<const std::uint32_t> indices, std::string debug_name) {
    const std::size_t vertex_count = vertices.size() / vertex_floats;
    if (vertex_count == 0 || indices.size() < 3 || vertices.size() % vertex_floats != 0) {
        return std::unexpected(gpu::make_error(gpu::GpuError::buffer_creation_failed,
                                               VK_ERROR_INITIALIZATION_FAILED,
                                               "bad mesh data for meshlet build"));
    }

    const std::size_t bound =
        meshopt_buildMeshletsBound(indices.size(), max_meshlet_vertices, max_meshlet_triangles);
    std::vector<meshopt_Meshlet> meshlets(bound);
    std::vector<std::uint32_t> meshlet_vertices(bound * max_meshlet_vertices);
    std::vector<std::uint8_t> meshlet_triangles(bound * max_meshlet_triangles * 3);

    const std::size_t meshlet_count = meshopt_buildMeshlets(
        meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(),
        indices.size(), vertices.data(), vertex_count, vertex_floats * sizeof(float),
        max_meshlet_vertices, max_meshlet_triangles, /*cone_weight=*/0.0F);
    meshlets.resize(meshlet_count);

    // Pack GPU meshlet descriptors (bounds for task-shader culling) and
    // triangle indices (3x uint8 local -> one uint32 per triangle).
    std::vector<GpuMeshlet> gpu_meshlets(meshlet_count);
    std::size_t packed_triangle_count = 0;
    for (const meshopt_Meshlet& m : meshlets) {
        packed_triangle_count += m.triangle_count;
    }
    std::vector<std::uint32_t> packed_triangles(packed_triangle_count);
    std::size_t packed_offset = 0;

    for (std::size_t i = 0; i < meshlet_count; ++i) {
        const meshopt_Meshlet& m = meshlets[i];
        const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            meshlet_vertices.data() + m.vertex_offset, meshlet_triangles.data() + m.triangle_offset,
            m.triangle_count, vertices.data(), vertex_count, vertex_floats * sizeof(float));

        GpuMeshlet& g = gpu_meshlets[i];
        g.vertex_offset = m.vertex_offset;
        g.triangle_offset = static_cast<std::uint32_t>(packed_offset);
        g.vertex_count = m.vertex_count;
        g.triangle_count = m.triangle_count;
        std::memcpy(g.center, bounds.center, sizeof(g.center));
        g.radius = bounds.radius;
        std::memcpy(g.cone_apex, bounds.cone_apex, sizeof(g.cone_apex));
        g.cone_cutoff = bounds.cone_cutoff;
        std::memcpy(g.cone_axis, bounds.cone_axis, sizeof(g.cone_axis));
        g.pad = 0.0F;

        for (std::uint32_t t = 0; t < m.triangle_count; ++t) {
            const std::uint8_t* tri = meshlet_triangles.data() + m.triangle_offset + t * 3;
            packed_triangles[packed_offset + t] = static_cast<std::uint32_t>(tri[0]) |
                                                  (static_cast<std::uint32_t>(tri[1]) << 8) |
                                                  (static_cast<std::uint32_t>(tri[2]) << 16);
        }
        packed_offset += m.triangle_count;
    }
    meshlet_vertices.resize(meshlets.back().vertex_offset + meshlets.back().vertex_count);

    MeshletSet out;
    auto make_buffer = [&](gpu::Buffer& dst, std::span<const std::byte> data,
                           const char* name) -> std::expected<void, gpu::Error> {
        auto buf = gpu::Buffer::create(allocator,
                                       {.size = data.size(),
                                        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        .host_visible = true,
                                        .debug_name = name});
        if (!buf) {
            return std::unexpected(buf.error());
        }
        if (!buf->write(data)) {
            return std::unexpected(gpu::make_error(gpu::GpuError::buffer_creation_failed,
                                                   VK_ERROR_INITIALIZATION_FAILED,
                                                   "meshlet buffer write failed"));
        }
        dst = std::move(*buf);
        return {};
    };
    auto bytes = [](const auto& v) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(v.data()),
                                          v.size() * sizeof(v[0]));
    };

    const std::vector<float> vertex_copy(vertices.begin(), vertices.end());
    if (auto r = make_buffer(out.vertices_, bytes(vertex_copy), (debug_name + "_vertices").c_str());
        !r) {
        return std::unexpected(r.error());
    }
    if (auto r = make_buffer(out.meshlets_, bytes(gpu_meshlets), (debug_name + "_meshlets").c_str());
        !r) {
        return std::unexpected(r.error());
    }
    if (auto r = make_buffer(out.vertex_indices_, bytes(meshlet_vertices),
                             (debug_name + "_vidx").c_str());
        !r) {
        return std::unexpected(r.error());
    }
    if (auto r = make_buffer(out.triangle_indices_, bytes(packed_triangles),
                             (debug_name + "_tidx").c_str());
        !r) {
        return std::unexpected(r.error());
    }

    out.meshlet_count_ = static_cast<std::uint32_t>(meshlet_count);
    out.vertex_count_ = static_cast<std::uint32_t>(vertex_count);
    return out;
}

} // namespace vulcanGE::render
