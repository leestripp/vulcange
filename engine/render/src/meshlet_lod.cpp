#include <vge/render/meshlet_lod.hpp>

#include <algorithm>
#include <cfloat>
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
constexpr std::size_t partition_target = 4;
constexpr std::uint32_t max_lod_levels = 8;

// One DAG cluster during the host build: a meshlet plus the global vertex
// indices of its triangles (needed for partition/simplify of parent groups).
struct HostCluster {
    std::vector<std::uint32_t> indices; // global vertex indices
    float group_error = 0.0F;           // error of the group this cluster belongs to
    float parent_error = 0.0F;          // error of the parent group
    std::uint32_t lod_level = 0;
};

// meshopt meshlet build for one index buffer; returns meshlet descriptors
// plus, for each meshlet, its global index list.
struct LevelBuild {
    std::vector<GpuMeshlet> meshlets;
    std::vector<std::uint32_t> vertex_indices; // global ids, max_meshlet_vertices per meshlet
    std::vector<std::uint32_t> packed_triangles;
    std::vector<std::vector<std::uint32_t>> cluster_indices;
};

[[nodiscard]] LevelBuild build_level(const float* vertices, std::size_t vertex_count,
                                     const std::uint32_t* indices, std::size_t index_count) {
    const std::size_t bound =
        meshopt_buildMeshletsBound(index_count, max_meshlet_vertices, max_meshlet_triangles);
    std::vector<meshopt_Meshlet> meshlets(bound);
    std::vector<std::uint32_t> meshlet_vertices(bound * max_meshlet_vertices);
    std::vector<std::uint8_t> meshlet_triangles(bound * max_meshlet_triangles * 3);

    const std::size_t meshlet_count = meshopt_buildMeshlets(
        meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
        vertices, vertex_count, vertex_floats * sizeof(float), max_meshlet_vertices,
        max_meshlet_triangles, /*cone_weight=*/0.0F);
    meshlets.resize(meshlet_count);

    LevelBuild out;
    out.meshlets.resize(meshlet_count);
    out.cluster_indices.resize(meshlet_count);

    std::size_t packed_total = 0;
    std::size_t vertex_total = 0;
    for (const meshopt_Meshlet& m : meshlets) {
        packed_total += m.triangle_count;
        vertex_total += m.vertex_count;
    }
    out.packed_triangles.resize(packed_total);
    out.vertex_indices.resize(vertex_total);

    std::size_t packed_offset = 0;
    std::size_t vertex_offset = 0;
    for (std::size_t i = 0; i < meshlet_count; ++i) {
        const meshopt_Meshlet& m = meshlets[i];
        const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            meshlet_vertices.data() + m.vertex_offset, meshlet_triangles.data() + m.triangle_offset,
            m.triangle_count, vertices, vertex_count, vertex_floats * sizeof(float));

        GpuMeshlet& g = out.meshlets[i];
        g.vertex_offset = static_cast<std::uint32_t>(vertex_offset);
        g.triangle_offset = static_cast<std::uint32_t>(packed_offset);
        g.vertex_count = m.vertex_count;
        g.triangle_count = m.triangle_count;
        std::memcpy(g.center, bounds.center, sizeof(g.center));
        g.radius = bounds.radius;
        std::memcpy(g.cone_apex, bounds.cone_apex, sizeof(g.cone_apex));
        g.cone_cutoff = bounds.cone_cutoff;
        std::memcpy(g.cone_axis, bounds.cone_axis, sizeof(g.cone_axis));
        g.pad = 0.0F;

        // Global vertex ids for this meshlet's local vertex slots.
        std::vector<std::uint32_t> local_to_global(m.vertex_count);
        for (std::uint32_t v = 0; v < m.vertex_count; ++v) {
            const std::uint32_t global = meshlet_vertices[m.vertex_offset + v];
            local_to_global[v] = global;
            out.vertex_indices[vertex_offset + v] = global;
        }
        for (std::uint32_t t = 0; t < m.triangle_count; ++t) {
            const std::size_t src = m.triangle_offset + t * 3;
            const std::uint32_t i0 = meshlet_triangles[src + 0];
            const std::uint32_t i1 = meshlet_triangles[src + 1];
            const std::uint32_t i2 = meshlet_triangles[src + 2];
            out.packed_triangles[packed_offset + t] = i0 | (i1 << 8) | (i2 << 16);
            auto& ci = out.cluster_indices[i];
            ci.push_back(local_to_global[i0]);
            ci.push_back(local_to_global[i1]);
            ci.push_back(local_to_global[i2]);
        }
        packed_offset += m.triangle_count;
        vertex_offset += m.vertex_count;
    }
    return out;
}

} // namespace

std::expected<MeshletLodTree, gpu::Error>
MeshletLodTree::create(const gpu::Allocator& allocator, std::span<const float> vertices,
                       std::span<const std::uint32_t> indices, std::string debug_name) {
    const std::size_t vertex_count = vertices.size() / vertex_floats;
    if (vertex_count == 0 || indices.size() < 3 || vertices.size() % vertex_floats != 0) {
        return std::unexpected(gpu::make_error(gpu::GpuError::buffer_creation_failed,
                                               VK_ERROR_INITIALIZATION_FAILED,
                                               "bad mesh data for meshlet LOD build"));
    }

    // ---- Level 0 ----
    LevelBuild base = build_level(vertices.data(), vertex_count, indices.data(), indices.size());
    std::vector<HostCluster> current;
    current.reserve(base.meshlets.size());
    for (auto& ci : base.cluster_indices) {
        current.push_back({.indices = std::move(ci)});
    }

    // All levels share one concatenated buffer set. all_clusters mirrors
    // all_meshlets one-to-one; level_base is the index where the level
    // currently being processed starts.
    std::vector<GpuMeshlet> all_meshlets = std::move(base.meshlets);
    std::vector<std::uint32_t> all_vindices = std::move(base.vertex_indices);
    std::vector<std::uint32_t> all_triangles = std::move(base.packed_triangles);
    std::vector<HostCluster> all_clusters;
    for (const auto& c [[maybe_unused]] : current) {
        all_clusters.push_back({.group_error = 0.0F, .parent_error = 0.0F, .lod_level = 0});
    }
    std::size_t level_base = 0;

    // ---- DAG levels ----
    std::uint32_t lod = 0;
    while (current.size() > 1 && lod < max_lod_levels) {
        // Partition this level's clusters into groups of ~4.
        std::vector<std::uint32_t> concat_indices;
        std::vector<std::uint32_t> counts;
        concat_indices.reserve(current.size() * 96);
        counts.reserve(current.size());
        for (const auto& c : current) {
            concat_indices.insert(concat_indices.end(), c.indices.begin(), c.indices.end());
            counts.push_back(static_cast<std::uint32_t>(c.indices.size()));
        }
        std::vector<std::uint32_t> group_of(current.size());
        const std::size_t group_count = meshopt_partitionClusters(
            group_of.data(), concat_indices.data(), concat_indices.size(), counts.data(),
            counts.size(), vertices.data(), vertex_count, vertex_floats * sizeof(float),
            partition_target);

        // Simplify each group into the parent level.
        std::vector<HostCluster> next;
        std::vector<float> group_errors(group_count, 0.0F);
        for (std::size_t g = 0; g < group_count; ++g) {
            std::vector<std::uint32_t> merged;
            float child_max = 0.0F;
            for (std::size_t c = 0; c < current.size(); ++c) {
                if (group_of[c] == g) {
                    merged.insert(merged.end(), current[c].indices.begin(),
                                  current[c].indices.end());
                    child_max = std::max(child_max, current[c].group_error);
                }
            }
            std::vector<std::uint32_t> simplified(merged.size());
            float error = 0.0F;
            const std::size_t simplified_count = meshopt_simplify(
                simplified.data(), merged.data(), merged.size(), vertices.data(), vertex_count,
                vertex_floats * sizeof(float), merged.size() / 2, FLT_MAX,
                meshopt_SimplifyLockBorder | meshopt_SimplifyErrorAbsolute, &error);
            simplified.resize(simplified_count);
            const float group_error = std::max(error, child_max);
            group_errors[g] = group_error;

            LevelBuild parent =
                build_level(vertices.data(), vertex_count, simplified.data(), simplified.size());
            // Offsets in parent.meshlets are level-local; rebase them onto
            // the concatenated all-levels index buffers.
            const std::uint32_t vbase = static_cast<std::uint32_t>(all_vindices.size());
            const std::uint32_t tbase = static_cast<std::uint32_t>(all_triangles.size());
            for (std::size_t i = 0; i < parent.meshlets.size(); ++i) {
                parent.meshlets[i].vertex_offset += vbase;
                parent.meshlets[i].triangle_offset += tbase;
                next.push_back({.indices = std::move(parent.cluster_indices[i]),
                                .group_error = group_error,
                                .parent_error = 0.0F, // set when the next level exists
                                .lod_level = lod + 1});
                all_meshlets.push_back(parent.meshlets[i]);
                all_clusters.push_back({.group_error = group_error,
                                        .parent_error = 0.0F,
                                        .lod_level = lod + 1});
            }
            all_vindices.insert(all_vindices.end(), parent.vertex_indices.begin(),
                                parent.vertex_indices.end());
            all_triangles.insert(all_triangles.end(), parent.packed_triangles.begin(),
                                 parent.packed_triangles.end());
        }

        // The level just processed now knows its parent groups' errors.
        for (std::size_t c = 0; c < current.size(); ++c) {
            all_clusters[level_base + c].parent_error = group_errors[group_of[c]];
        }

        if (next.size() >= current.size()) {
            // No reduction (topology bound): the new level cannot go further.
            current = std::move(next);
            level_base = all_clusters.size() - current.size();
            for (std::size_t c = 0; c < current.size(); ++c) {
                all_clusters[level_base + c].parent_error = FLT_MAX;
            }
            ++lod;
            break;
        }
        current = std::move(next);
        level_base = all_clusters.size() - current.size();
        ++lod;
    }

    // Last level clusters are DAG roots: no parent above them.
    for (std::size_t c = 0; c < current.size(); ++c) {
        all_clusters[level_base + c].parent_error = FLT_MAX;
    }
    const std::uint32_t levels = lod + 1;

    // ---- LOD info buffer ----
    std::vector<std::uint32_t> lod_info(all_clusters.size() * lod_info_uints);
    for (std::size_t i = 0; i < all_clusters.size(); ++i) {
        std::memcpy(&lod_info[i * lod_info_uints + 0], &all_clusters[i].group_error, 4);
        std::memcpy(&lod_info[i * lod_info_uints + 1], &all_clusters[i].parent_error, 4);
        lod_info[i * lod_info_uints + 2] = all_clusters[i].lod_level;
        lod_info[i * lod_info_uints + 3] = 0;
    }

    // ---- Upload ----
    auto upload = [&](const void* data, std::size_t size,
                      const char* name) -> std::expected<gpu::Buffer, gpu::Error> {
        auto buf = gpu::Buffer::create(allocator, {.size = size,
                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   .host_visible = true,
                                                   .debug_name = name});
        if (!buf) {
            return std::unexpected(buf.error());
        }
        auto up = buf->write(
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(data), size));
        if (!up) {
            return std::unexpected(gpu::make_error(gpu::GpuError::buffer_creation_failed,
                                                   VK_ERROR_INITIALIZATION_FAILED,
                                                   "meshlet LOD buffer upload failed"));
        }
        return std::move(*buf);
    };

    MeshletLodTree tree;
    auto meshlets = upload(all_meshlets.data(), all_meshlets.size() * sizeof(GpuMeshlet),
                           (debug_name + "_meshlets").c_str());
    if (!meshlets) return std::unexpected(meshlets.error());
    tree.meshlets_ = std::move(*meshlets);
    auto verts = upload(vertices.data(), vertices.size() * sizeof(float),
                        (debug_name + "_verts").c_str());
    if (!verts) return std::unexpected(verts.error());
    tree.vertices_ = std::move(*verts);
    auto vidx = upload(all_vindices.data(), all_vindices.size() * sizeof(std::uint32_t),
                       (debug_name + "_vidx").c_str());
    if (!vidx) return std::unexpected(vidx.error());
    tree.vindices_ = std::move(*vidx);
    auto tidx = upload(all_triangles.data(), all_triangles.size() * sizeof(std::uint32_t),
                       (debug_name + "_tidx").c_str());
    if (!tidx) return std::unexpected(tidx.error());
    tree.tindices_ = std::move(*tidx);
    auto linfo = upload(lod_info.data(), lod_info.size() * sizeof(std::uint32_t),
                        (debug_name + "_lodinfo").c_str());
    if (!linfo) return std::unexpected(linfo.error());
    tree.lod_info_ = std::move(*linfo);

    tree.cluster_count_ = static_cast<std::uint32_t>(all_clusters.size());
    tree.vertex_count_ = static_cast<std::uint32_t>(vertex_count);
    tree.lod_levels_ = levels;
    return tree;
}

} // namespace vulcanGE::render
