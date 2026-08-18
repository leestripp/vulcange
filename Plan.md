# vulcanGE: Next-Gen C++23 Vulkan Game Engine Architecture Plan

## 1. Executive Summary & Core Philosophy

This document outlines the architectural specification for **vulcanGE**, a high-performance, data-oriented game engine built natively for Linux with C++23 and Vulkan 1.3+.

### 1.1 Core Principles

- **Modern C++23 First (Linux Native):** Target modern GCC 13+ and Clang 16+ toolchains on Linux. Leverage C++23 modules, `std::expected`, `std::optional`, `std::mdspan`, monadic operations, coroutines, and explicit memory safety patterns.
- **GPU-Driven Modern Vulkan Architecture:** Utilize Vulkan 1.3 features—Dynamic Rendering, Mesh Shaders (`VK_EXT_mesh_shader`), Descriptor Buffers / Bindless Texturing, dynamic state, and GPU-driven rendering (`ExecuteIndirect` & Hi-Z occlusion culling) on Linux Vulkan drivers (Mesa/RADV, NVIDIA, ANV).
- **Deferred-First Render Pipeline:** Standard G-Buffer layout optimized for high dynamic lighting counts with clustered deferred / compute-tiled shading.
- **Fiber/Task Graph Concurrency:** Lock-free, work-stealing job system utilizing C++20/23 coroutines and POSIX/Linux threading primitives (`std::jthread`, lock-free queues) for async task scheduling and high-throughput async file I/O (io_uring compatible).
- **Data-Oriented Design (DOD):** Cache-friendly component layouts (Archetype/Sparse-set ECS) with tight cache locality for transform updates, physics, and skinning matrices.
- **Strict Cross-Platform / Linux Focus:** Zero platform lock-in. No Windows SDKs, DirectX components, or MSVC-specific extensions. All subsystem libraries build natively on GCC/Clang on Linux.
- **Single-Player Scope:** Online multiplayer/networking is explicitly out of scope for v1. The deterministic fixed-timestep simulation and decoupled event architecture keep the door open for future netcode without re-architecting.

## 2. Technology Stack & Modern C++23 Features

| Subsystem | Technology / Feature | C++23 / API Usage | Linux & Build Tooling |
|---|---|---|---|
| Target OS Platform | Linux (x86_64 & ARM64) | POSIX / Linux System V ABI | Native Linux kernel primitives |
| Language & Compilers | C++23 Standard | Modules (`export module engine;`), `std::expected<T, E>`, `std::mdspan`, `std::print`, `std::generator` | GCC 13+, Clang 16+ (Ninja / CMake) |
| Graphics API | Vulkan 1.3+ | Dynamic Rendering, Bindless Descriptors, Mesh Shaders, Synchronization2 | LunarG Vulkan SDK for Linux, Mesa / RADV / NVIDIA |
| Windowing & Input | GLFW 3.4+ (locked) | Native Wayland & X11 surface integration, gamepad input | Linux native display backends |
| Tasking / Async | Custom Fiber Task System | Coroutines (`co_await`, `co_return`), `std::jthread`, `std::stop_token`, atomic primitives | pthread / LinuxFutex / io_uring async I/O |
| Memory Management | Custom PMR Allocators | `std::pmr::memory_resource`, Lock-free Frame Allocators, Pool/Arena Allocators | POSIX mmap / standard PMR / VMA |
| Linear Algebra Math | Custom `vulcanGE::math` | C++23 concepts, SIMD intrinsics (AVX2/NEON), native Vulkan coordinates | Native C++23 module |
| Physics Subsystem | Jolt Physics | Multi-threaded solver via `JPH::JobSystem` C++23 coroutine bridge | Cross-platform C++17/20, native Linux build (MIT) |
| Asset Pipeline | glTF 2.0 (fastgltf / cgltf) | `std::expected` for file loading, meshlet construction (meshoptimizer), zstd pack compression | Fully open-source cross-platform C++ libs |
| 2D & Gameplay UI | Custom MSDF (msdfgen) / ImGui | Dynamic 2D dynamic batching, MSDF text atlas, vector clipping | Cross-platform native C++ render pipelines |
| Spatial Audio | Miniaudio / PipeWire Integration | HRTF 3D positioning, async sound streaming, sub-mix buses | Linux ALSA / PulseAudio / PipeWire backends |
| Profiling & Diagnostics | Tracy Profiler / VMA | GPU timestamp queries, lock-free trace zones, memory tracking | Native Linux tracing primitives |

### 2.1 C++23 System Integration Highlights

#### Monadic Error Handling & Memory Safety

Replace traditional exception overhead and standard error codes with monadic `std::expected<T, EngineError>`:

```cpp
// Example C++23 Vulkan initialization signature pattern in vulcanGE
[[nodiscard]] auto create_vulkan_image(const ImageCreateInfo& info)
    -> std::expected<VulkanImage, RenderError>;
```

#### Multi-dimensional Array Abstraction (`std::mdspan`)

Utilize `std::mdspan` for zero-cost multi-dimensional data access over flat GPU staging buffers, heightmaps, and volumetric sky textures:

```cpp
// 3D slice access into volumetric atmosphere lookup tables
auto sky_lut_view = std::mdspan(staging_buffer.data(), dim_x, dim_y, dim_z);
```

#### Safety & Resource Lifetime (RAII & Smart Handles)

Enforce explicit handle types (`StrongHandle<T>`) for GPU resources to prevent use-after-free and double-free bugs. Modern object management through move-only resource wrappers with explicit destruction validation.

## 3. Linear Algebra & Matrix Math Architecture (`vulcanGE::math`)

### 3.1 Matrix Library Strategy: Evaluation & Recommendation

Selecting the linear algebra library for vulcanGE requires balancing build times, modern C++23 features, SIMD throughput, permissive licensing, and native Vulkan coordinate space alignment.

| Metric / Feature | GLM | HLSL++ | DirectXMath | Custom `vulcanGE::math` (Selected) |
|---|---|---|---|---|
| License | MIT | MIT | MIT | Proprietary/Internal (Engine License) |
| Vulkan Coordinates | Legacy OpenGL Default (Requires Macros) | Configurable | DirectX Default | Native Vulkan Baked-In ($Y$-Down, $Z \in [0, 1]$) |
| SIMD Support | Optional / Limited | SSE/AVX2/NEON | SSE2/AVX2/NEON | Native AVX2 / AVX-512 / ARM NEON Intrinsics |
| C++23 Integration | Legacy Templates (C++11/14) | Modern C++17/20 | C++14 / MSVC Legacy | Full C++23 Modules (`std::simd`/`std::concepts`) |
| Compile-Time Impact | Heavy Template Overhead | Fast | Moderate | Instant (Zero Header Bloat) |
| Memory Alignment | Unaligned Default | 16-byte Aligned | 16-byte Aligned | Cache-Line & 16/32-Byte SIMD Aligned |

#### Architectural Decision: Custom C++23 Math Engine (`vulcanGE::math`)

While GLM is widely used, it suffers from significant compile-time overhead, non-SIMD default layouts, and OpenGL-centric assumptions that require messy global macros (`GLM_FORCE_DEPTH_ZERO_TO_ONE`, `GLM_FORCE_LEFT_HANDED`).

vulcanGE will implement a custom, lightweight, header-module SIMD math library (`vulcanGE::math`) tailored specifically for modern C++23 and Vulkan:

- **Zero External Dependencies:** Native compile time is instantaneous using C++23 modules (`import vulcanGE.math;`).
- **Explicit SIMD Vectorization:** Direct AVX2 and ARM NEON intrinsics for $4 \times 4$ matrix operations, quaternion rotations, and SIMD frustum plane culling.
- **Baked-in Vulkan Coordinate System:** Eliminates runtime matrix inverse conversions or macro hacks.

### 3.2 Vulkan vs. OpenGL Coordinate Conventions

Vulkan differs fundamentally from legacy OpenGL in clip space, coordinate origin, and projection matrix bounds:

```text
[ OpenGL Clip Space ]                      [ Vulkan Clip Space ]
     +Y (Up)                                     -Y (Up)
      ▲                                           │
      │   +Z (Out of Screen)                      │   +Z (Into Screen)
      │  ↗                                        ▼  ↙
─── ── ┼ ──── ───► +X                     ─── ── ┼ ──── ───► +X
      │                                           │
      │                                           ▼
                                                 +Y (Down)

  Z Depth: [-1.0, +1.0]                     Z Depth: [0.0, +1.0]
```

**Normalized Device Coordinates (NDC) Y-Axis:**

- OpenGL: $Y$-axis points UP (Bottom-Left origin).
- Vulkan: $Y$-axis points DOWN (Top-Left origin).

**Depth Buffer Range ($Z$-Axis Clip):**

- OpenGL: $[-1.0, +1.0]$ (Symmetric NDC depth).
- Vulkan: $[0.0, +1.0]$ (Zero-to-One NDC depth range).

**Handedness & View Space:**

vulcanGE adopts a Right-Handed View Space ($+X$ right, $+Y$ up, $+Z$ back towards viewer) and projects into Vulkan's Left-Handed NDC ($+Y$ down, $Z \in [0, 1]$).

### 3.3 Projection Strategy: Reverse-Z Perspective & Orthographic

#### Reverse-Z Infinite Perspective (Default)

Standard depth mapping ($Z \in [0, 1]$ with a finite far plane) wastes most depth-buffer precision near the camera and forces arbitrary far-plane clipping of sky and distant terrain. vulcanGE uses **reversed depth with an infinite far plane**: the near plane maps to depth 1.0 and infinity maps to 0.0, giving near-uniform precision across the entire view distance.

For vertical FOV $\theta_{\text{fov}}$, aspect ratio $a = \frac{w}{h}$, and near plane $z_n$ (with $z_f \to \infty$):

$$
\mathbf{P}_{\text{ReverseZ}} = \begin{bmatrix}
\frac{1}{a \cdot \tan\left(\frac{\theta_{\text{fov}}}{2}\right)} & 0 & 0 & 0 \\
0 & -\frac{1}{\tan\left(\frac{\theta_{\text{fov}}}{2}\right)} & 0 & 0 \\
0 & 0 & 0 & z_n \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

The negative sign in the second row flips $Y$ natively for Vulkan clip space; the third row maps reversed depth into $[0.0, 1.0]$.

**Reverse-Z Pipeline Rules (global invariants, enforced engine-wide):**

- Depth buffer format: `VK_FORMAT_D32_SFLOAT` — floating-point depth is mandatory for reverse-Z.
- Clear depth to **0.0**; all depth tests use `VK_COMPARE_OP_GREATER_OR_EQUAL` (inverted from legacy convention).
- Hi-Z pyramids store **minimum** depth per texel instead of maximum.
- A finite-far variant (`perspective_vulkan_finite`) is retained for tools and validation only.

#### Orthographic Projection (Shadow Cascades, Tools, 2D UI)

For an orthographic view volume $[l, r] \times [b, t] \times [z_n, z_f]$ with reverse-Z depth:

$$
\mathbf{P}_{\text{Ortho}} = \begin{bmatrix}
\frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\
0 & \frac{2}{b-t} & 0 & -\frac{b+t}{b-t} \\
0 & 0 & \frac{1}{z_f - z_n} & \frac{z_f}{z_f - z_n} \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

#### Vulkan Dynamic Viewport Alternative

Alternatively, Vulkan 1.3 allows dynamic viewport flipping without altering standard projection matrices by setting a negative viewport height:

```cpp
VkViewport viewport{};
viewport.x = 0.0f;
viewport.y = static_cast<float>(framebuffer_height); // Start at bottom
viewport.width = static_cast<float>(framebuffer_width);
viewport.height = -static_cast<float>(framebuffer_height); // Negative height flips Y
viewport.minDepth = 0.0f;
viewport.maxDepth = 1.0f;
vkCmdSetViewport(command_buffer, 0, 1, &viewport);
```

### 3.4 C++23 Math Subsystem Code Layout

```cpp
// vulcanGE Math Subsystem Layout (vulcanGE::math)
namespace vulcanGE::math {

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

struct alignas(16) Vector4 {
    union {
        struct { float x, y, z, w; };
        #if defined(__AVX2__)
        __m128 simd_reg;
        #endif
    };

    constexpr Vector4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    constexpr Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};

struct alignas(64) Matrix4x4 {
    Vector4 cols[4];

    // Native Vulkan Reverse-Z Perspective (Infinite Far Plane, see 3.3)
    [[nodiscard]] static constexpr Matrix4x4 perspective_vulkan(
        float fov_y_rad, float aspect, float z_near) noexcept
    {
        Matrix4x4 result{};
        const float focal_length = 1.0f / std::tan(fov_y_rad * 0.5f);

        result.cols[0].x = focal_length / aspect;
        result.cols[1].y = -focal_length; // Y-Flip for Vulkan Top-Left Clip Space
        result.cols[2].z = 0.0f;          // Reverse-Z: infinite far plane
        result.cols[2].w = -1.0f;
        result.cols[3].z = z_near;        // Near plane -> depth 1.0, infinity -> depth 0.0
        result.cols[3].w = 0.0f;

        return result;
    }
};

} // namespace vulcanGE::math
```

## 4. Core Engine Architecture & Scene Graph

```text
                                  +-----------------------+
                                  |   vulcanGE Core &     |
                                  |   Module Subsystems   |
                                  +-----------+-----------+
                                              |
                +-----------------------------+-----------------------------+
                |                             |                             |
      +---------v---------+         +---------v---------+         +---------v---------+
      | Task Graph System |         |  Memory System    |         | Data-Oriented ECS |
      | Coroutines/Fibers |         | PMR/Arena/Pool    |         | & Scene Hierarchy |
      +---------+---------+---------+---------+---------+---------+---------+---------+
                |                             |                             |
 +--------------v-----------------------------v-----------------------------v--------------+
 |                                  vulcanGE Engine Pipeline Layer                          |
 +-------+--------------------+--------------------+-------------------+--------------------+
         |                    |                    |                   |
 +-------v-------+    +-------v-------+    +-------v-------+   +-------v-------+
 | Vulkan 1.3    |    | Character     |    | Terrain &     |   | Physics Engine|
 | Deferred      |    | Bone          |    | Dynamic Sky   |   | Solver (Jolt) |
 | Render Graph  |    | Animation     |    | Engine        |   | Integration   |
 +---------------+    +---------------+    +---------------+   +---------------+
```

### 4.1 Task System & Fiber Architecture

- **Lock-free Work-Stealing Pool:** Worker threads pinned to hardware threads (`std::jthread` with `pthread_setaffinity_np` on Linux).
- **Coroutine-driven Tasks:** `co_await TaskGraph::YieldTo(Priority::High)` to eliminate thread-blocking delays during GPU transfers or asset deserialization.
- **Fiber Stack Management:** Low-overhead context switching for long-running processes (e.g., dynamic terrain quadtree evaluations).

### 4.2 Data-Oriented Scene Graph Architecture

To ensure maximum cache locality and bypass traditional object-oriented pointer chasing, the vulcanGE Scene Graph is decoupled into a **Flat Hierarchy Layout** synced directly with the ECS.

```text
+---------------------------------------------------------------------------------+
|                              Flat Transform Memory                              |
|  [ Parent Indices ] :  [-1,   0,   0,   1,   1,   2,   -1,   6]                 |
|  [ Local Matrices ] :  [ M0,  M1,  M2,  M3,  M4,  M5,  M6,  M7]                 |
|  [ World Matrices ] :  [ W0,  W1,  W2,  W3,  W4,  W5,  W6,  W7]                 |
+---------------------------------------------------------------------------------+
```

**Key Design Aspects:**

- **Contiguous Flat Hierarchy Arrays:**
  - Parent-child relationships are represented using dynamic integer index vectors (`std::vector<int32_t> parent_indices`).
  - Nodes are sorted topologically so that parent transforms always precede child transforms in memory array order.
- **Single-pass SIMD vectorization** updates global world matrices without recursive tree traversal:

$$
\mathbf{W}_i = \mathbf{W}_{\text{parent}[i]} \cdot \mathbf{L}_i
$$

- **Dynamic Dirty-Mask Tracking:**
  - Bitsets track dirty hierarchy nodes; when a parent node shifts, an SIMD dirty propagation mask updates affected downstream continuous slices.
- **Spatial Indexing & Culling Layer:**
  - **Loose Octree & Bounded Volume Hierarchy (BVH):** Static geometry is stored in a BVH for fast raycasts; dynamic entities are tracked in a coarse Loose Octree to accelerate frustum and occlusion queries before passing data to Vulkan `ExecuteIndirect`.

### 4.3 ECS Core Architecture

- **Archetype-Based Storage:** Components are grouped by archetype signature into contiguous 16 KiB SoA chunks for vectorized iteration; a sparse-set indirection maps `EntityId -> (archetype, row)` for O(1) random access.
- **Entity Lifecycle:** Generational entity IDs (32-bit index + 32-bit generation) prevent stale-handle use-after-free; destruction is deferred and flushed at frame end.
- **System Scheduler:** Systems declare component read/write access up front; the task graph builds a dependency DAG per frame and runs non-conflicting systems in parallel on worker fibers.
- **Queries & Change Detection:** Cached archetype queries with per-component change ticks (`Added<T>`, `Modified<T>`) feed the renderer's dirty-mask propagation and the test harness's determinism checks.

### 4.4 Reflection, Serialization & Save Games

C++23 ships without reflection, so vulcanGE implements a compile-time reflection layer:

- **Code-Generated Reflection:** A CMake-driven generator parses annotated engine headers and emits registration code into a module unit — no runtime RTTI, no external dependencies. Designed to migrate to C++26 static reflection (`std::meta`) once toolchains mature.
- **Binary Save Format:** Versioned, endian-stable snapshots of ECS archetype chunks with schema migration handlers; save/load streams asynchronously through the VFS / io_uring path (§12.1) without hitching gameplay.
- **Editor Bridge:** Blender custom properties (§13.2) are validated against the same reflection schema, guaranteeing cooked `.vpack` component payloads never drift from runtime types.
- **Determinism Hooks:** Fixed-seed RNG and ordered Event Bus delivery make save + input replay bit-identical — the foundation of the local test harness (§16.4).

### 4.5 Main Loop, Game Clock & Time Dilation

```text
[ Poll Input (GLFW) ] -> [ Fixed Simulation Ticks ] -> [ Variable Update ]
      -> [ Task Graph Flush ] -> [ Frame Graph Record/Submit ] -> [ Present ]
```

- **Fixed Simulation Rate:** Gameplay logic ticks at 60 Hz (configurable) using the same accumulator pattern as physics (§11.4); rendering interpolates between ticks.
- **Game Clock:** A central `GameClock` exposes scaled time (slow-mo), pause (zero simulation dt while UI keeps rendering), and frame counters for deterministic replay.
- **Fixed Frame Order:** Input sampling -> event flush -> transform update -> physics sync -> render record is contractually ordered each frame; subsystem code may not reorder it.

### 4.6 Camera System & Floating Origin (Camera-Relative Rendering)

- **Camera Stack:** ECS camera components (FOV, exposure, post-process settings) resolve into one active camera per viewport; supports smooth blending, shake impulses, and split-screen viewports.
- **Floating Origin:** World positions are stored as 64-bit integer cell + 32-bit float local offset. When the camera crosses a cell boundary (~8 km), the scene graph rebases around the camera — eliminating floating-point jitter in large terrain worlds.
- **Camera-Relative Rendering:** All view-space matrices are computed relative to the camera cell each frame; the GPU never sees absolute world coordinates.

### 4.7 Input System Architecture (GLFW)

- **GLFW 3.4+ Exclusive:** Windowing, keyboard/mouse, and joystick/gamepad input via GLFW (gamepad mappings database) on both Wayland and X11 backends. SDL3 was evaluated and rejected — GLFW keeps the dependency surface minimal and is sufficient with the action-mapping layer below.
- **Action Mapping Layer:** Raw device events map to named gameplay actions (`Jump`, `Fire`, `Interact`) through rebindable, context-sensitive maps (Gameplay / Menu / Vehicle contexts with priority stacking).
- **Input Recording:** Every input event is captured with a frame timestamp into a deterministic stream, feeding the replay/test harness (§16.4).

## 5. Vulkan 1.3+ Modern Deferred Render Pipeline

### 5.1 Frame Graph Architecture

The renderer centers around a dynamic **Frame Graph** (Render Graph) that automatically manages pipeline barriers (`VkMemoryBarrier2`), resource aliasing, dynamic pass scheduling, and dynamic rendering scope (`vkCmdBeginRendering`).

### 5.2 Pipeline Stages & Pass Breakdown

```text
[ Pass 1: Depth Prepass / Hi-Z Generation ]
       │
       ▼
[ Pass 2: G-Buffer Pass (Mesh Shaders / Bindless Material ID) ]
       │  ├── Target 0: World Normal (RGB10A2_UNORM)
       │  ├── Target 1: Albedo / Roughness (RGBA8_UNORM)
       │  ├── Target 2: Metallic / AO / Material Parameters (RGBA8_UNORM)
       │  └── Target 3: Motion Vectors (RG16_SFLOAT)
       ▼
[ Pass 3: Physics Debug & Dynamic Shadow Pass (Cascaded Shadow Maps / Virtual Shadow Maps) ]
       │
       ▼
[ Pass 4: Clustered Deferred Lighting Pass (Compute Driven) ]
       │  ├── Atmospheric Scattering / Dynamic Sky Blend
       │  ├── Dynamic Diffuse GI (DDGI Probes / SSR)
       │  └── Local Lights (Punctual, Area Lights) Evaluation
       ▼
[ Pass 5: Dynamic Terrain & Foliage Render Pass ]
       │
       ▼
[ Pass 6: Forward+ Transparent Pass / Volumetric Clouds ]
       │
       ▼
[ Pass 7: Post-Processing Stack (TAA, Motion Blur, DoF, Bloom, Tone Map) ]
       │
       ▼
[ Pass 8: 2D HUD / Gameplay UI / Debug Render Pass (SDF Text & Dynamic Batching) ]
```

### 5.3 GPU-Driven Geometry Pipeline

- **Mesh Shader Pipeline:** Task shaders execute cluster-level culling (frustum, backface, cone-culling). Mesh shaders generate primitives directly on the GPU.
- **Bindless Resource Architecture:** Global descriptor indexing. Textures are bound once into global descriptor arrays; material IDs are passed through push constants:

$$
\text{Index} = \text{MaterialPushConstant.texture\_index}
$$

- **Hi-Z Occlusion Culling:** Two-phase occlusion culling using the previous frame's Hierarchical Z-Buffer (Hi-Z).

### 5.4 Swapchain, Presentation & Frame Pacing

- **Present Modes:** `VK_PRESENT_MODE_FIFO_KHR` (vsync) by default, `MAILBOX` for low latency, `IMMEDIATE` for benchmarking — runtime-selectable via cvars.
- **Frames in Flight:** Double-buffered acquire/submit with per-frame fences and semaphore recycling; `VK_KHR_present_id` / `VK_KHR_present_wait` used for pacing telemetry where available.
- **Resize & Suboptimal Handling:** Swapchain recreation is driven through the Frame Graph (transient resource rebinding) without stalling the task system.
- **HDR Output:** `VK_COLOR_SPACE_HDR10_ST2084_EXT` and scRGB swapchain formats with configurable paper-white nits. Wayland HDR uses the merged `color-management-v1` protocol — native on Mesa 25.1+ under KWin Plasma 6.x (AMD/Intel). The NVIDIA proprietary driver lacks Wayland color-management support, so NVIDIA users fall back to a Gamescope HDR session; the engine ships a `--gamescope` launch wrapper for this path.
- **VRR:** Adaptive sync via mailbox presentation and compositor negotiation on Wayland/X11.

### 5.5 Queue Families, Async Compute & Transfer Streaming

- **Queue Selection:** Dedicated graphics queue, async compute queue (VSM page updates, DDGI probes, particles), and a dedicated transfer queue for the io_uring -> staging -> GPU streaming path (§12.1).
- **Overlap:** The Frame Graph schedules independent passes onto compute and graphics queues concurrently (e.g., VSM page updates overlapping the G-Buffer pass), synchronized via Synchronization2 events.
- **Async Readback:** A readback ring handles screenshots, occlusion query results, and test-harness frame capture without pipeline stalls.

### 5.6 GPU Feature Matrix & Fallback Paths

| Feature | Primary Path | Fallback Path | Notes |
|---|---|---|---|
| Mesh Shaders | `VK_EXT_mesh_shader` task/mesh pipeline | Classic vertex pipeline + GPU `ExecuteIndirect` culling | RADV & ANV since Mesa 22.3; NVIDIA 515+; NVK experimental only (Mesa 26.2) — unsupported tier |
| Bindless | `VK_EXT_descriptor_buffer` | `VK_EXT_descriptor_indexing` sets | RADV/NVIDIA ship both; ANV coverage varies |
| Occlusion | Hi-Z two-phase GPU culling | Frustum-only culling | |
| Ray Tracing | Optional `VK_KHR_ray_tracing_pipeline` (DDGI probe rays) | Compute-probe DDGI (§6.6) | RT is never a hard requirement |
| Minimum Spec | Vulkan 1.3, 8 GB VRAM, AVX2 CPU | — | AMD RDNA2+ (RADV Mesa 23.1+), NVIDIA GTX 16 / RTX 20+ (driver 535+), Intel Arc (ANV Mesa 23.1+) |

### 5.7 GPU Particle & VFX System

- **Compute-Simulated Particles:** Particle state (position, velocity, age) lives in SSBOs; a compute pass integrates forces, depth-buffer collisions, and curl-noise turbulence.
- **Indirect Rendering:** Alive-particle compaction writes indirect draw counts — zero CPU readback, single `vkCmdDrawIndirect` per effect batch.
- **Effect Graph:** Data-driven effect definitions (emitters, curves, bursts) authored as JSON, cooked from Blender particle/geometry-node exports, instanced via the Event Bus.
- **Lighting Integration:** Lit particles sample the clustered light grid (§6); a separate unlit additive pass covers sparks and embers.

### 5.8 Temporal Upscaling & Dynamic Resolution Scaling

- **FSR 2/3 & XeSS:** Pluggable upscaler interface; FSR 2 is the default (open source, native Linux Vulkan). DLSS is excluded (proprietary, poor Wayland support).
- **Dynamic Resolution:** A GPU-frame-time-driven resolution controller (0.5x–1.0x) keeps frame times in budget, reacting within ~3 frames.
- **AA Order:** TAA at render resolution -> upscale -> optional FXAA post pass.

## 6. Advanced Dynamic Lighting, GI & Virtual Shadow Maps

### 6.1 Shadowing Architecture: Virtual Shadow Maps (VSM) & CSM

**Virtual Shadow Maps (VSM):**

- Multi-resolution physical cache page table ($16384 \times 16384$ virtual resolution mapped into $4096 \times 4096$ physical memory allocations).
- Direct integration with Mesh Shader cull stages to render only invalidated physical pages when lights or objects shift.

**Cascaded Shadow Maps (CSM) with PCSS:**

- Fallback directional shadow pipeline utilizing 4 logarithmically spaced depth cascades with Percentage-Closer Soft Shadows (PCSS) filter kernels for realistic contact-hardening shadows.

### 6.2 Global Illumination & Reflection Pipeline

**Dynamic Diffuse Global Illumination (DDGI Probes):**

- Grid of ray-traced / compute-probed irradiance and distance spheres ($16 \times 8 \times 16$).
- Probes store $16 \times 16$ octahedral-mapped irradiance to eliminate indirect light leaking around geometry walls.

**Screen-Space Reflections (SSR) with Hi-Z Tracing:**

- Compute-based cone-tracing through the Hierarchical Z-Buffer for fast specular reflections, blended smoothly with cubemap probe fallbacks.

### 6.3 Ambient Occlusion (GTAO)

Ground-Truth Ambient Occlusion computed at half resolution from G-Buffer normals and the Hi-Z pyramid, temporally accumulated with the TAA history, and applied multiplicatively in the deferred lighting pass. Baked per-material AO textures (G-Buffer target 2) modulate the result.

### 6.4 Volumetric Fog & Light Shafts

A frustum-aligned froxel volume ($160 \times 90 \times 64$) integrates participating-media density, height-fog falloff, and shadowed in-scattering from the clustered light grid and VSM/CSM. Temporal reprojection stabilizes the volume; designers place local density volumes in Blender as fog box components.

### 6.5 Deferred Decal System

Screen-space decal volumes render after the G-Buffer pass, re-projecting albedo/normal/roughness into G-Buffer targets before lighting. Decal culling shares the clustered light data structure; supports bullet holes, tire marks, and projected texture mods.

### 6.6 Ray Tracing Stance

Hardware ray tracing (`VK_KHR_ray_tracing_pipeline`) is **optional, never required**. Where present it accelerates DDGI probe ray updates and future RT reflections; compute-probe DDGI and SSR are the guaranteed baseline on all supported Linux drivers.

## 7. Dynamic Mesh LOD & Dynamic Tessellation System

### 7.1 Distance & Screen-Space Error Metrics

vulcanGE handles Level of Detail (LOD) dynamically using screen-space geometric projection error metrics, preventing geometry popping and reducing primitive counts at long distances.

**Screen-Space Geometric Error Formula**

Let $\delta$ be the world-space bounding geometric error metric of a meshlet/LOD cluster, $d$ be the distance from the camera, $h_{\text{screen}}$ be the viewport pixel height, and $\theta_{\text{fov}}$ be the vertical field of view angle. The screen-space error $E_{\text{screen}}$ in pixels is given by:

$$
E_{\text{screen}} = \frac{\delta \cdot h_{\text{screen}}}{2 \cdot d \cdot \tan\left(\frac{\theta_{\text{fov}}}{2}\right)}
$$

If $E_{\text{screen}} > E_{\text{threshold}}$, the meshlet/LOD switches to a higher resolution level.

### 7.2 Meshlet LOD Cluster DAG & GPU Tessellation

- **Meshlet Cluster Hierarchy (Nanite-Style DAG):** Pre-computed meshlet clusters (64 vertices, 128 triangles max per cluster via meshoptimizer) grouped into hierarchical Directed Acyclic Graphs (DAGs).
- **Task Shaders** evaluate DAG nodes in parallel on the GPU to select the appropriate LOD per cluster seamlessly without CPU intervention.
- **GPU Hardware Dynamic Tessellation & Compute Mesh Displacement:** Dynamic Tessellation for high-fidelity surfaces (water, dynamic terrain heightmaps, detailed brick displacement). Fallback using dynamic Mesh Shaders for meshlet subdivision where hardware tessellation stages (`VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT`) are bypassed for custom geometry generation.
- **Seamless Cross-Fading:** Dynamic stochastic dithering / temporal alpha cross-fading in the fragment shader to eliminate temporal pop artifacts between discrete LOD levels.

## 8. Dynamic Atmospheric Sky System

### 8.1 Physical Sky Model

The dynamic sky engine is based on physically based atmospheric scattering (Rayleigh & Mie scattering) combined with a volumetric cloud pipeline:

**Rayleigh Scattering Strength:**

$$
\beta_R(\lambda) = \frac{8\pi^3 (n^2 - 1)^2}{3 N e_s} \frac{1}{\lambda^4}
$$

**Mie Scattering Phase Function (Henyey-Greenstein):**

$$
P_{HG}(\theta, g) = \frac{1 - g^2}{4\pi (1 + g^2 - 2g \cos\theta)^{3/2}}
$$

### 8.2 Dynamic Sky Components

**Precomputed Atmospheric Scattering Lookup Tables (LUTs):**

- Transmittance LUT ($256 \times 64$).
- Multiple Scattering LUT ($32 \times 32$).
- Sky View LUT ($192 \times 108$) updated per frame for dynamic sun position.

**Volumetric Cloud System:**

- Compute shader raymarching through 3D Perlin-Worley noise fields.
- Temporal reprojection for performance stabilization.

**Day/Night Transition Manager:**

- Dynamic celestial body simulation (Sun, Moon positions, Star map rotation).

## 9. Camera-Distance Terrain & Dynamic Vegetation System

### 9.1 Dynamic Camera-Distance Terrain System

The terrain subsystem operates completely GPU-driven using camera-centric distance evaluation coupled with dynamic quadtree LOD decomposition.

```text
       +-------------------------------------------------------+
       | Camera Position (P_cam) & View Frustum State          |
       +---------------------------+---------------------------+
                                   |
                                   ▼
       +-------------------------------------------------------+
       | GPU Task Shader / Quadtree Chunk Evaluation           |
       |  - Calculate Distance: d = ||P_cam - P_chunk||        |
       |  - Evaluate LOD: Level = clamp(f(d, error), 0, max)   |
       +---------------------------+---------------------------+
                                   |
              +--------------------+--------------------+
              |                                         |
              ▼                                         ▼
   [ Mesh Shader Meshlet Pass ]              [ Compute Displacement Pass ]
   (Generate Terrain Surface)                (Sampler Heightmap LUT)
```

#### 1. Distance-Based Quadtree Node Splitting

The terrain ground is divided into a dynamic quadtree centered around the active camera position. Quadtree nodes split or merge based on the continuous camera distance function:

$$
\text{LOD}_{\text{level}} = \text{clamp}\left( \left\lfloor -\log_2 \left( \frac{\Vert{} \mathbf{p}_{\text{camera}} - \mathbf{p}_{\text{chunk}} \Vert{}}{R_{\text{base}}} \right) \right\rfloor, 0, \text{LOD}_{\text{max}} \right)
$$

#### 2. GPU Heightmap Tessellation & Virtual Texturing

- **Mesh Shaders** construct dynamic terrain patches while sampling heightmaps using bicubic filtering.
- **Runtime Virtual Texturing (RVT):** Blend up to 16 distinct surface layers (e.g., rock, mud, snow, grass) into a continuous physical cache texture atlas ($8192 \times 8192$) based on terrain slope, altitude, and normal orientation.

### 9.2 Camera-Driven Dynamic Vegetation Engine

The vegetation pipeline renders millions of grass blades, bushes, and trees using compute-shader instance generation filtered by camera distance.

```text
+---------------------------------------------------------------------------------+
| vulcanGE Dynamic Vegetation Pipeline                                            |
|                                                                                 |
| [ Density Maps & Terrain Quadtree ] ──► [ GPU Compute Spatial Hash Grid ]       |
|                                                     │                           |
|                                                     ▼                           |
| [ Dynamic Multi-Stage Culling ] ◄─── [ 1. Distance Culling  (d > d_max) ]       |
|   │                                  [ 2. Frustum Culling (AABB vs Planes)]      |
|   │                                  [ 3. Hi-Z Occlusion Culling Buffer ]       |
|   ▼                                                                             |
| [ GPU Indirect Draw Buffers ] ────► [ Vulkan vkCmdDrawIndexedIndirectCount ]    |
+---------------------------------------------------------------------------------+
```

**Vegetation Pipeline Steps:**

- **Procedural Placement & Spatial Hash Grid:**
  - Instance generation is driven procedurally via dynamic GPU noise patterns and RVT vegetation density maps.
  - Instance data (Transform, Species ID, Color Variance) is managed in SSBO arrays mapped across spatial grid cells.
- **Multi-Stage GPU Instance Culling:**
  - **Distance Culling:** Instances beyond $d_{\text{max}}$ (e.g., $150\,\text{m}$ for fine grass, $2000\,\text{m}$ for trees) are culled instantly.
  - **Frustum Culling:** Axis-Aligned Bounding Boxes (AABBs) checked against camera view frustum planes.
  - **Hi-Z Occlusion Culling:** Culled against depth pyramid to prevent drawing foliage behind hills or buildings.
- **GPU ExecuteIndirect Commands:**
  - Compute shaders write remaining valid instance counts into an indirect parameter buffer (`VkDrawIndexedIndirectCommand`).
  - Main rendering pass renders all instances in a single `vkCmdDrawIndexedIndirectCount` call.
- **Wind Vector Simulation & Dynamic LOD Fading:**
  - Foliage stems deform in real time using 2D Directional Perlin Wind Noise fields.
  - Vegetation transitions between LODs (High-poly Mesh $\rightarrow$ Low-poly Billboard) using dithered alpha stippling based on distance $d$.

## 10. Skeletal & Character Bone Animation System

### 10.1 Architecture & Pipeline

The animation engine evaluates poses using SIMD-accelerated math and offloads skinning updates to compute shaders.

```text
       [ Animation Graph / State Machine (CPU Tasks) ]
                              │
                              ▼
        [ Blending Trees, IK Solvers (FABRIK / Two-Bone) ]
                              │
                              ▼
        [ SIMD Local Transform Evaluation (SoA Layout) ]
                              │
                              ▼
    [ Compute Shader Skinning Buffer (Dual Quaternion / Linear) ]
                              │
                              ▼
    [ Mesh Shader Pipeline (Render Characters with Pose Matrices) ]
```

### 10.2 Bone Skinning Formulations

**Linear Blend Skinning (LBS):**

$$
v' = \sum_{i=1}^{k} w_i \mathbf{M}_i v
$$

Where $v$ is the vertex position, $w_i$ is the bone weight, and $\mathbf{M}_i$ is the bone matrix transformation.

**Dual Quaternion Skinning (DQS):** Used for preserving volume at joint bends (eliminating "candy-wrapper" artifacts).

**GPU Instanced Character Skinning:** Storing matrix palettes in SSBOs (Shader Storage Buffer Objects) mapped directly via bindless descriptors.

### 10.3 Animation Production Features

- **Root Motion:** Root-bone deltas extracted per clip can drive the physics capsule (KCC) directly for grounded, deterministic locomotion.
- **Animation Events/Notifies:** Time-marked triggers (footsteps, hit frames) dispatched through the Event Bus to audio, physics, and gameplay.
- **Clip Compression:** 48-bit quaternion quantization + linear key reduction baked at cook time; clips stream per-LOD through the VFS.
- **Blend Spaces & Retargeting:** 1D/2D locomotion blend spaces; skeleton retargeting via bind-pose remap tables authored in Blender.

## 11. Physics Engine Integration & Selection Research

### 11.1 Physics Engine Candidate Evaluation

To power vulcanGE's physics subsystem natively on Linux, candidates were evaluated against five strict engine criteria:

- **Permissive Commercial License:** Must allow royalty-free commercial distribution (MIT, BSD-3-Clause, Zlib).
- **Cross-Platform Linux Native:** Zero proprietary binary blobs or Windows-only OS dependencies. Clean build with GCC 13+ and Clang 16+.
- **Multi-Threaded Job System Architecture:** Clean abstraction to interface with vulcanGE's coroutine fiber task graph.
- **Data-Oriented SIMD Performance:** Vectorization (AVX2/AVX-512/NEON) scaling across high CPU core counts.
- **Comprehensive Feature Set:** Dynamic rigid bodies, Continuous Collision Detection (CCD), Kinematic Character Controllers (KCC), ragdolls, and broadphase raycasting.

| Metric / Feature | Jolt Physics | NVIDIA PhysX 5 | Bullet Physics 3 | ReactPhysics3D |
|---|---|---|---|---|
| License | MIT | BSD-3-Clause (CPU) | Zlib | Zlib |
| Linux Native Support | 100% Native (GCC/Clang) | Partial (GPU Binary Lock) | 100% Native | 100% Native |
| C++ Modernity | C++17/20/23 Modern | C++14/17 | Legacy C++03/11 | Modern C++17 |
| Task Graph Integration | Exceptional (`JPH::JobSystem`) | Moderate / Verbose | Poor | Limited |
| Multi-Core Scaling | Exceptional | High | Moderate | Single-threaded core |
| SIMD Support | AVX2 / AVX-512 / NEON | AVX2 / GPU (CUDA) | SSE / NEON | Limited |
| Character Controller | Virtual & Rigid KCC | Full KCC | Basic Character | Custom Required |
| Ragdoll & IK Support | Native Ragdoll Driven | Full Ragdoll | Basic | Manual |
| Footprint & Complexity | Compact & Modular | Heavy & Monolithic | Moderate | Lightweight |

### 11.2 Selection Decision: Jolt Physics (MIT)

**Jolt Physics** was selected as the core physics provider for vulcanGE on Linux due to its purpose-built multi-threaded job interface, production AAA track record (Horizon Forbidden West), modern C++ architecture, zero proprietary GPU lock-in, and 100% native Linux GCC/Clang compatibility under the permissive MIT license.

### 11.3 Task Graph Bridge Implementation (`JPH::JobSystem`)

Jolt allows custom task dispatch by deriving from `JPH::JobSystem`. We map Jolt solver tasks directly onto vulcanGE's fiber task graph (`vulcanGE::async`):

```cpp
namespace vulcanGE::physics {

class JoltJobSystemBridge final : public JPH::JobSystem {
public:
    JoltJobSystemBridge(uint32_t max_jobs, uint32_t max_barriers)
        : JPH::JobSystem(max_jobs, max_barriers) {}

    // Enqueue job directly into vulcanGE's C++23 coroutine Task Graph
    virtual void QueueJob(JPH::JobSystem::Job* job) override {
        vulcanGE::async::dispatch_task([job]() {
            job->Execute();
            job->Release();
        });
    }

    virtual void QueueJobs(JPH::JobSystem::Job** jobs, uint32_t num_jobs) override {
        for (uint32_t i = 0; i < num_jobs; ++i) {
            QueueJob(jobs[i]);
        }
    }
};

} // namespace vulcanGE::physics
```

### 11.4 Scene Graph Synchronization & Timestep Interpolation

- **Contiguous Transform Sync:** Active dynamic body transforms produced by Jolt write directly into vulcanGE's flat contiguous scene transform buffer.
- **Fixed Timestep Accumulator Loop:** Physics ticks execute at fixed intervals ($60\,\text{Hz}$ or $120\,\text{Hz}$) decoupled from rendering frame rates. Visual rendering poses interpolate using factor $\alpha$:

$$
\mathbf{x}_{\text{render}} = \alpha \mathbf{x}_{\text{current}} + (1 - \alpha) \mathbf{x}_{\text{previous}}
$$

### 11.5 Subsystem Capabilities

- **Rigid Body Solver:** Static terrain colliders, dynamic rigid bodies, compound shapes.
- **Kinematic Character Controller (KCC):** Virtual and rigid-body character controllers with slope handling, step projection, and collision response.
- **Ragdoll & Animated Physics Integration:** Transition between skeletal animation poses and dynamic physics ragdoll states using joint constraints.

### 11.6 Vehicles, Soft Bodies & Cloth

Jolt's `VehicleConstraint` (wheeled/tracked) and soft-body/cloth solver are exposed through the same `JoltJobSystemBridge`. Destruction uses pre-fractured rigid-body clusters with impulse-activated constraints — all simulated on the fiber task graph without main-thread cost.

## 12. Asset Pipeline, VFS & Dynamic Shader System

### 12.1 Virtual File System (VFS) & Linux io_uring Async I/O

To achieve instantaneous zero-hitch loading on modern Linux NVMe storage, vulcanGE implements a high-performance Virtual File System (VFS):

- **Kernel High-Throughput Storage (io_uring):** Offload asynchronous read requests directly to the Linux kernel via non-blocking ring submission queues.
- **Memory-Mapped Package Containers (.vpack):** Game assets are packaged into custom aligned binary blobs mounted via POSIX `mmap()` for zero-copy GPU memory staging.

### 12.2 Optimized Binary Formats & Texture Compression

- **Custom Geometry Container (.vmesh):** Pre-processed meshlet metadata, cluster bounds, and SIMD-aligned vertex arrays generated offline via meshoptimizer.
- **Texture Supercompression (Basis Universal / BC7):** Textures compressed to Basis Universal / BC7 formats offline. Transcoded asynchronously directly into GPU dynamic descriptor arrays.
- **Package Compression (zstd):** `.vpack` blocks are zstd-compressed (high cook-time level, dictionary-trained per project) and decompressed on worker fibers inside io_uring completion handling.

### 12.3 Shader Live Hot-Reloading Pipeline

- **Dynamic SPIR-V Runtime Compilation:** Integrated shaderc / DXC library wrappers enable live monitoring of GLSL / HLSL shader sources.
- **Hot Pipeline Re-creation:** When a modified shader file is saved, the dynamic Frame Graph re-creates affected `VkPipeline` handles in the background without restarting the engine.

### 12.4 Engine Material Binary (.vmat)

`.vmat` is the cooked material format referenced by the Blender pipeline (§13): shader variant ID, PBR parameter block, bindless texture indices, and render-state flags (blend mode, double-sided, decal layer). Cooked from Blender Principled BSDF graphs and hot-reloadable at runtime exactly like shaders (§12.3).

### 12.5 Runtime Streaming, Memory Budgets & GPU Residency

- **Streaming Scheduler:** A priority queue (distance x screen size x staleness) drives async loads over io_uring; chunk manifests from Blender streaming cells (§13.4) define load/unload triggers.
- **Memory Budgets:** Per-heap budgets enforced via VMA statistics; over-budget conditions trigger LRU eviction of texture mips and mesh LODs — never allocation failure.
- **Texture Mip Streaming:** Basis/BC7 textures load low mips first; high mips stream in from visibility feedback.
- **GPU Residency:** Bindless descriptor arrays use sparse backing where supported; eviction defragments during load transitions.

## 13. Editor & Authoring Pipeline: Blender as the Engine Editor

**Blender is the designated editor and primary authoring environment for vulcanGE.** All world building, scene composition, asset placement, level design, animation setup, and gameplay entity configuration are authored inside Blender and exported into the engine's runtime formats. Blender is not linked into the runtime; it is the external, standalone editor toolchain for the engine.

### 13.1 Blender-Centric Editor Philosophy

- **Single Source of Truth:** The `.blend` file is the authoritative level/scene document for a vulcanGE project. Engine packages (`.vpack`) and runtime binaries (`.vmesh`, `.vmat`) are derived from Blender source assets.
- **Open Source & Linux Native:** Blender runs natively on Linux, aligns with the engine's open-source ethos, and supports Python automation—ideal for a custom export and validation pipeline.
- **Familiar Authoring Surface:** Artists, designers, and level builders use Blender's native 3D viewport, outliner, collections, modifiers, and animation tools for content creation.
- **Runtime-Agnostic Editing:** The editor provides an immediate-mode, WYSIWYG-like authoring experience while the engine executes the optimized, compiled runtime data in a separate game process.

### 13.2 Scene-to-Engine Data Mapping

Blender scene constructs are mapped directly into vulcanGE runtime data structures:

| Blender Construct | vulcanGE Runtime Equivalent | Notes |
|---|---|---|
| Collection | Prefab / Asset Bundle | Maps to a loadable entity hierarchy and asset group. |
| Empty / Object | ECS Entity + Transform | Empties become locators, spawners, or trigger volumes. |
| Mesh Object | Static / Skinned Mesh Renderable | Exported as `.vmesh` with meshlet metadata. |
| Armature / Pose | Skeleton & Animation Clip | Driven into the SIMD pose evaluator and GPU skinning buffer. |
| Material | Engine Material Template | Converted to PBR parameters and bindless texture indices. |
| Light (Sun/Spot/Point/Area) | Renderable Light Component | Clustered deferred light data with physical units. |
| Camera | Camera Component | Viewport, projection, and post-process settings. |
| Collection Instance | Prefab Instantiation | Supports nested prefabs and reusable environment modules. |
| Custom Properties | Engine Component Data | Exported as typed ECS component payloads. |
| Curve / Lattice | Spline / Deform Path | Optional runtime curves for rails, rivers, or wind fields. |

### 13.3 Export Pipeline

The Blender-to-engine export pipeline is implemented as a Blender Add-on written in Python:

```text
[ .blend Scene / Level / Prefab ]
              │
              ▼
[ vulcanGE Blender Add-on (Python) ]
              │
              ├── Validate scene naming, material usage, and mesh topology
              ├── Bake meshlets, LOD chains, and cluster bounds (meshoptimizer)
              ├── Bake animation clips, bone palettes, and skinning weights
              ├── Bake light parameters, camera settings, and transform hierarchy
              ├── Generate RVT / vegetation density masks from vertex paint
              └── Write engine metadata, UUIDs, and dependency manifest
              │
              ▼
[ Intermediate glTF 2.0 + .vge.json Metadata ]
              │
              ▼
[ Offline Cook Tool (Linux CLI, C++23) ]
              │
              ├── Transcode textures to Basis Universal / BC7
              ├── Compile SPIR-V variants and material shader graphs
              ├── Pack geometry into .vmesh / .vpack containers
              └── Produce final .vpack runtime asset bundle
              │
              ▼
[ Engine Runtime loads .vpack via VFS + io_uring ]
```

### 13.4 Custom Blender Add-on Features

- **Engine Properties Panel:** Per-object, per-material, and per-scene panels to attach engine-specific data (tags, collision layers, gameplay scripts, spawn rules, LOD settings).
- **Viewport Preview Mode:** Optional integration with the engine's Vulkan renderer to render a live preview inside Blender's viewport (via a shared offscreen swap or screenshot comparison workflow).
- **Scene Validation:** Real-time checks for non-manifold geometry, missing UVs, oversized textures, and unsupported material nodes before export.
- **Hot Export & Live Reload:** One-click export to the running engine; the VFS watches the cooked `.vpack` directory and reloads changed assets without restarting the engine process.
- **Level Streaming Chunks:** Designer can mark collections as streaming cells; the add-on writes chunk manifests and camera-distance activation rules.

### 13.5 Workflow Summary

1. Designers and artists create content natively in Blender.
2. The vulcanGE Add-on validates and exports scenes to glTF 2.0 + engine metadata.
3. The Linux CLI cook tool compiles and packages the runtime asset bundle.
4. The engine loads the bundle through the VFS and renders the authored world using the Vulkan pipeline.
5. Iteration is accelerated by hot-reload, keeping the editor and engine in sync during development.

### 13.6 Coordinate & Unit Conversion (Blender -> vulcanGE)

- **Units:** Both Blender (default configuration) and glTF use **meters — 1 unit = 1 m on both sides**. No unit scaling is applied anywhere in the pipeline; the add-on asserts `scene.unit_settings.scale_length == 1.0` during validation.
- **Axes:** Blender is right-handed **Z-up**; glTF and vulcanGE world space are right-handed **Y-up**. Conversion is a single -90° rotation about +X, applied once by the add-on at export (glTF `+Y up` convention), so runtime data is always Y-up — no runtime conversion cost and no double transforms.
- **Validation:** The add-on rejects exports with unapplied object transforms (location/rotation/scale not applied) so axis or scale bugs never enter cooked assets.

## 14. Spatial Audio Subsystem Architecture

### 14.1 Audio Subsystem Strategy & Integration

The audio subsystem is built on **Miniaudio** (permissive C/C++ MIT/Unlicense) interfacing natively with Linux audio backends (PipeWire, PulseAudio, and ALSA).

### 14.2 Spatial Audio Pipeline & Geometry Occlusion

```text
[ Spatial Sound Emitters (ECS Components) ]
                    │
                    ▼
[ HRTF Spatializer & Dynamic Distance Attenuation ]
                    │
                    ▼
[ Acoustic Raytracing (Physics Geometry Occlusion Queries) ]
                    │
                    ▼
[ Sub-mix Audio Buses (Master / Music / SFX / Voice) ] ──► [ Linux Backend Engine ]
```

- **HRTF (Head-Related Transfer Function):** Full 3D directional positioning using binaural HRTF filters for spatial immersion.
- **Physics-Driven Acoustic Occlusion:** Raycasts executed via Jolt Physics calculate sound path obstruction between listener and emitters, dynamically dampening higher frequencies (low-pass filtering) when occluded by terrain or structures.

### 14.3 Mixing, Buses & Ducking

A sub-mix bus graph (Master -> Music / SFX / Voice / UI) provides per-bus gain, low-pass filtering, and sidechain ducking (voice ducks SFX/music). Snapshot mixes are triggered from gameplay via the Event Bus (underwater, pause menu, low health).

## 15. 2D Gameplay UI, HUD & Text Rendering Subsystem

### 15.1 Dual-UI Architecture: Developer Tools vs. Gameplay UI

To prevent developer debug tooling from compromising gameplay UI rendering performance, vulcanGE segregates user interfaces into two decoupled pipelines:

```text
                                  [ Engine UI Trigger Event ]
                                               │
                       +-----------------------+-----------------------+
                       │                                               │
                       ▼                                               ▼
         [ 1. Gameplay HUD / Runtime UI ]               [ 2. Developer Debug Overlay ]
        (Multi-channel SDF Text & 2D Quad Batcher)     (Dear ImGui Docking Dashboard Engine)
                       │                                               │
                       ▼                                               ▼
         [ Dynamic Render Pass 8 Buffer ]               [ Dynamic Render Pass 8 Buffer ]
                       │                                               │
                       +-----------------------+-----------------------+
                                               │
                                               ▼
                               [ Vulkan Dynamic Scissor & Swapchain ]
```

- **Developer / Debug UI Layer (Dear ImGui Docking):** Reserved strictly for diagnostic tools, scene inspectors, memory allocators, and profiler zones. Evaluated as immediate-mode state.
- **Gameplay UI & HUD Layer (Dynamic 2D Batching & MSDF Text):** Retained-mode / Data-oriented ECS UI hierarchy optimized for resolution-independent crosshairs, health bars, dynamic minimaps, inventory overlays, settings menus, and localized text.

### 15.2 Multi-Channel Signed Distance Field (MSDF) Text Engine

Traditional rasterized font atlases degrade and become blurry under sharp dynamic scalings or fractional display DPI targets. vulcanGE utilizes Multi-Channel Signed Distance Fields (MSDF) for crisp text rendering across all screen resolutions.

**MSDF Texel Distance Evaluation Formula**

For a 3-channel MSDF texture $(r, g, b)$, the sharp edge distance metric $d$ at UV coordinate $(u, v)$ is calculated in the fragment shader via median filtering:

$$
d = \text{median}(r, g, b)
$$

The screen-space antialiased alpha value $\alpha_{\text{glyph}}$ is derived using the standard derivative magnitude:

$$
\alpha_{\text{glyph}} = \text{clamp}\left( \frac{d - 0.5}{\Vert{} \nabla d \Vert{}} + 0.5, 0.0, 1.0 \right)
$$

**Text Subsystem Architecture:**

- **Glyph Atlas Generator:** Pre-processed or dynamic runtime glyph atlas creation supporting full UTF-8 / Unicode internationalization.
- **Layout & Kerning Engine:** Fast text layout calculator handling kerning pairs, line wrapping, text justification, and dynamic text drop shadows / outlines inside fragment shaders without extra geometry.

### 15.3 High-Performance 2D Dynamic Renderer & Vector Clipping

- **Vulkan 2D Batch Renderer:** Single dynamic vertex array object (`VkBuffer` mapped staging memory) accumulating 2D dynamic primitives (textured quads, rounded rects, health radial fills, menu borders).
- **Hardware Dynamic Scissor Clipping (`vkCmdSetScissor`):** Efficient nested rectangular clipping for scrollable settings menus, UI containers, and dynamic HUD windows without requiring expensive stencil buffer clears.
- **Data-Driven HUD Widgets & Layout Nodes:** Custom flexbox-style 2D layout solver mapped directly onto ECS components. RmlUi was evaluated and **rejected**: its retained-mode HTML/CSS DOM duplicates the engine's own 2D batcher + MSDF text stack, has no native MSDF glyph support, and adds a large dependency for marginal authoring gain.

### 15.4 Localization & String Tables

All user-facing strings resolve through UTF-8 string tables (key -> per-locale text) loaded via the VFS. The MSDF atlas generator (§15.2) builds glyph coverage per active locale with fallback font stacks for CJK. String literals in gameplay code are rejected by a lint pass.

## 16. Diagnostics, Profiling & Developer Tooling Architecture

### 16.1 Embedded Real-time Profiling

- **Tracy Profiler Integration:** High-frequency C++ lock-free zone profiling across worker fiber threads.
- **Vulkan GPU Timestamp Profiling (`vkCmdWriteTimestamp2`):** Measure exact hardware execution duration per Frame Graph pass on GPU queues.

### 16.2 Memory Tracking & Debug UI Overlay

- **Vulkan Memory Allocator (VMA) Integration:** Track dynamic GPU VRAM allocation budgets, heap fragmentation, and dynamic staging buffers.
- **Docking Debug Dashboard (Dear ImGui):** Integrated developer UI overlay displaying real-time task thread utilization, entity inspectors, shadow cascade visualizations, and performance frame graphs.

### 16.3 Vulkan Validation & Wayland-Native GPU Debugging

**RenderDoc is not used** — it has no reliable Wayland capture support. vulcanGE's GPU debugging stack is instead:

- **Validation Layers:** `VK_LAYER_KHRONOS_validation` enabled in all development builds, plus GPU-Assisted Validation (GPU-AV) and synchronization validation in the test harness.
- **Debug Annotations:** `VK_EXT_debug_utils` labels and markers on every Frame Graph pass, resource, and queue submission — readable in validation output and in the in-engine tooling.
- **In-Engine Frame Inspector:** Built-in capture mode dumping per-pass images, G-Buffer visualization, overdraw and meshlet-cluster heatmaps, and live pipeline state into the debug dashboard (§16.2). This is the primary RenderDoc replacement and works natively on Wayland.
- **GFXReconstruct:** `gfxrecon-record` captures for binary-level Vulkan call replay and driver-bug isolation.
- **Crash Resilience:** `VK_EXT_robustness2` plus device-lost handling with automatic context recreation and a crash minidump (§16.5).

### 16.4 Local-Only Testing Strategy & Engine Test Hooks

**No CI — all testing runs locally on developer machines.** The engine is built test-first, with first-class hooks:

- **Unit Tests:** doctest (header-only) compiled into a `vge_tests` binary; math (SIMD vs scalar cross-validation), ECS, reflection, and serialization round-trips run on every local build via CTest.
- **Headless Test Mode:** `--headless` runs the engine without a swapchain (offscreen images only), enabling deterministic frame execution on any machine.
- **Golden-Image Regression:** The headless harness renders canonical test scenes and compares against checked-in reference images with a perceptual threshold, reporting diffs — the local replacement for CI render tests.
- **Deterministic Replay:** Fixed-seed RNG + recorded input streams (§4.7) replay exact gameplay sessions for regression hunting.
- **Sanitizer Builds:** Preset configurations for ASan, UBSan, and TSan (`cmake --preset asan` etc.); the full suite must pass clean under each.
- **In-Game Test Console:** Dev console commands (`test_run`, `test_scene`, `test_benchmark`) execute suites inside a running engine for interactive debugging.

### 16.5 Logging, Assertions & Crash Reporting

- **Logging:** Compile-time-leveled structured logging (spdlog-backed, async sinks) with per-subsystem channels; `std::print` is reserved for CLI tools.
- **Assertions:** `VGE_ASSERT` (debug only) and `VGE_VERIFY` (always on for I/O and security boundaries); fatal asserts route through the crash handler, never silent UB.
- **Crash Reporting:** A signal-handler minidump writer (self-contained, breakpad-style) captures stack traces, GPU device state, and the last N log lines; dumps land in `~/.local/share/vulcanGE/crashes/`.

## 17. Scripting, Event Bus & Dynamic Engine Plugins

### 17.1 C++23 Dynamic Plugin System (Dynamic Engine Modules)

Game modules are compiled as shared dynamic libraries (`.so` on Linux).

- **Hot-Reloadable Game Logic:** Dynamic loading using POSIX `dlopen()` / `dlsym()` allows dynamic updating of gameplay systems during engine execution.

### 17.2 Type-Safe Coroutine Event Bus

Decoupled messaging network (`EventBus::subscribe<T>()`) utilizing C++23 concepts and monadic handlers.

- Events can be listened to asynchronously within coroutines (`co_await EventBus::wait_for<CollisionEvent>()`).

### 17.3 Plugin ABI Stability Boundary

Hot-reloaded `.so` plugins must survive compiler upgrades, so every engine<->plugin call crosses a **versioned C ABI** (a `VGE_PLUGIN_API_VERSION`-tagged struct of function pointers). C++23 types never cross the boundary; plugins receive opaque handles plus reflection accessors (§4.4). `dlclose` is only issued after a full task-graph quiesce to prevent dangling fiber references.

### 17.4 Gameplay Code: Native C++23 (No Scripting VM)

vulcanGE ships **no scripting language** — this is a deliberate, locked decision. All gameplay code is written in native C++23 and compiled as hot-reloadable plugins (§17.1, §17.3): one language, one debugger (gdb/lldb), full Tracy profiling coverage, zero binding layer, and no VM overhead or GC pauses. Designer-facing behavior is data-driven instead: gameplay parameters live in reflected component payloads (§4.4) authored in Blender (§13.4) and tunable live through cvars (§17.5).

### 17.5 Configuration, Console Variables & Developer Console

- **Config Files:** TOML project/user configuration (`~/.config/vulcanGE/`) for graphics, audio, and input bindings, validated against the reflection schema with safe defaults.
- **Console Variables (cvars):** Every tunable (`r_*` render, `t_*` terrain, `log_*`) is a registered cvar settable at runtime from the dev console, config files, or the command line.
- **Dev Console:** Quake-style overlay console (built on §15 UI) with autocomplete, history, and the test-harness commands (§16.4).

## 18. Linux Build Environment & Implementation Roadmap

### 18.1 Linux Native Environment Setup

- **Build System:** CMake 3.28+ with Ninja generator.
- **Compilers:** GCC 13+ (`-std=c++23`) or Clang 16+ (`-std=c++2b`).
- **Packaging / Dependencies (locked):** vcpkg **manifest mode with a pinned baseline** for all third-party libraries (GLFW, Jolt, meshoptimizer, fastgltf, spdlog, zstd, msdfgen, Basis Universal) — identical dependency versions on every developer machine, which substitutes for the absent CI. System packages are used only for platform-level components vcpkg must not own: the Vulkan loader/SDK headers, `libwayland-dev`, `libx11-dev`, and Mesa/NVIDIA drivers.
- **C++23 Modules Fallback:** Modules are used where toolchain support is solid, but every module unit must also compile in a header-based configuration (CMake target switch). `import std;` in particular requires newer toolchains (GCC 15 / Clang 18+) and is gated behind a feature check — core architecture never hard-depends on modules.

### 18.2 Phased Roadmap

#### Phase 1: Linux Foundation, Math & Vulkan Bootstrap (Weeks 1–4)

- Setup GCC 13 / Clang 16 C++23 Linux build environment (CMake + Ninja).
- Implement modern C++23 SIMD Math Subsystem (`vulcanGE::math`) with Vulkan perspective projections ($Y$-down, $Z \in [0, 1]$).
- Build lock-free task system with coroutine support (`vulcanGE::async`) on Linux threading API.
- Implement Vulkan 1.3 Context initialization on Wayland/X11 with VMA.
- Implement Frame Graph infrastructure and flat memory Scene Graph architecture.

- Decide the reverse-Z depth convention (§3.3) and build the swapchain/presentation layer (§5.4) before any render pass work begins.
- Stand up validation layers, the in-engine test hooks (`--headless`, doctest/CTest), and a minimal direct-glTF asset path so early renderer work has real content before the cook pipeline exists.

#### Phase 2: Deferred Renderer, Material/LOD & Shadow System (Weeks 5–8)

- Build G-Buffer generation pass and dynamic pipeline creation.
- Implement Bindless Descriptor Management system.
- Implement Meshlet Cluster LOD system using meshoptimizer & Task/Mesh Shader pipelines.
- Integrate Virtual Shadow Maps (VSM) and Cascaded Shadow Maps (CSM).
- Build Clustered Compute Deferred Light Pass & Post-Processing Pipeline.

- Implement GTAO, the deferred decal pass, FSR 2 upscaling + dynamic resolution, and the GPU particle/VFX foundation.

#### Phase 3: Dynamic Sky, Camera-Distance Terrain & Vegetation (Weeks 9–12)

- Implement Atmospheric Scattering precomputation (Rayleigh/Mie LUTs).
- Build Volumetric Cloud Compute Pass.
- Develop Distance-based Quadtree Terrain System with RVT triplanar blending.
- Build Compute-driven Vegetation System with multi-stage culling and wind simulation.

- Extend the particle system with lit particles and weather effects (rain/snow through the froxel fog volume).

#### Phase 4: Animation, Physics, Audio, 2D HUD, Blender Editor & Developer Tooling (Weeks 13–16)

- Integrate Jolt Physics engine into `vulcanGE::async` task graph via `JoltJobSystemBridge`.
- Build glTF 2.0 animated mesh importer using fastgltf.
- Implement SIMD Pose Evaluation and GPU Compute Skinning.
- Implement MSDF Text Engine & Dynamic 2D Batch UI Render Pass for HUD and menus.
- Integrate Spatial Audio (Miniaudio/PipeWire) with physics acoustic occlusion.
- Integrate Tracy Profiler, VMA tracking, and ImGui debug overlay dashboard.
- Setup io_uring Linux VFS asset loading and shader dynamic hot-reloading.
- **Develop the vulcanGE Blender Add-on, glTF 2.0 scene export, and `.vpack` cook pipeline so Blender serves as the engine editor.**

- Implement reflection-driven save/load, localization string tables, the settings/cvar system, dev console, and crash reporting.
- Establish the hot-reload gameplay plugin workflow (§17.1/§17.3): live C++23 code swap without engine restart, exercised end-to-end by the test harness.

## 19. Summary Verification Checklist

| Subsystem | Core Requirement | Verification Method |
|---|---|---|
| Linux Target | 100% Linux Native | Builds with GCC 13+ / Clang 16+ on Linux without non-portable libraries |
| Language | Full C++23 compliance | Compile with `-std=c++23` and zero compiler warnings (`-Wall -Wextra -Werror`) |
| Math Engine | Vulkan-native `vulcanGE::math` | Validate $Y$-down projection matrices and SIMD alignment across AVX2/NEON |
| Scene Graph | Data-Oriented Flat Layout | Benchmark continuous memory transform traversal without pointer chasing |
| Renderer | Deferred Vulkan 1.3+ Pipeline | Validate G-Buffer execution on Mesa/RADV or NVIDIA Linux drivers using the in-engine frame inspector and validation layer output |
| Shadows & GI | Virtual Shadow Maps / DDGI | Verify page physical cache allocations and probe irradiance accuracy |
| LOD System | Dynamic Meshlet DAG / Screen-Space Error | Measure triangle count reduction when adjusting distance & FOV thresholds |
| Atmosphere | Physically Based Dynamic Sky | Verify sun angle raymarch updates in real-time |
| Terrain | Distance Quadtree / Dynamic RVT | Test LOD transitions seamlessly without visual popping |
| Vegetation | GPU Indirect Dynamic Instancing | Render $1,000,000+$ grass/tree instances at $>60\,\text{FPS}$ |
| Animation | Multi-threaded Skeletal Pipeline | Benchmarking 100+ instanced animated characters at $>60\,\text{FPS}$ |
| Physics | Multi-threaded Rigid Body & KCC | Validate sweep-and-slide collision accuracy under heavy load in Jolt Physics |
| 2D HUD / Text | MSDF Text & Dynamic 2D Batcher | Validate crisp text rendering across resolution changes with hardware scissor clipping |
| Asset VFS | io_uring Async Loading | Verify non-blocking asset streaming directly from NVMe to GPU staging buffers |
| Audio | Spatial HRTF & Raytraced Occlusion | Validate dynamic low-pass sound dampening behind physics barriers |
| Diagnostics | Tracy & GPU Timestamps | Capture full-frame CPU fiber thread zones and GPU render pass timings |
| Presentation | Swapchain / HDR / VRR | Validate present modes, HDR10 output, and resize handling under Wayland and X11 |
| Input | GLFW Action Mapping | Verify rebindable actions, gamepad hotplug, and input-replay determinism |
| Save System | Reflection-Based Serialization | Round-trip a 10,000-entity world save/load with zero data drift |
| Particles | GPU Compute VFX | Simulate and render 100,000+ particles at $>60\,\text{FPS}$ with zero CPU readback |
| Floating Origin | Camera-Relative Rendering | Verify zero vertex jitter at 100 km world coordinates |
| Test Harness | Headless Golden-Image Tests | All canonical scenes pass perceptual diff locally under ASan/UBSan/TSan |
| Blender Editor | Blender as Engine Editor | Verify `.blend` scene exports correctly into `.vpack` and renders identically in engine |
