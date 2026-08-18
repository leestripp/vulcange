add_library(vge_gpu STATIC
    src/instance.cpp
    src/device.cpp
    src/swapchain.cpp
    src/allocator.cpp
    src/buffer.cpp
    src/image.cpp
    src/command.cpp
    src/sync.cpp
    src/pipeline.cpp
)
add_library(vge::gpu ALIAS vge_gpu)
target_include_directories(vge_gpu PUBLIC include)
find_package(Vulkan REQUIRED)
target_link_libraries(vge_gpu PUBLIC Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator PRIVATE vge_warnings)
target_compile_features(vge_gpu PUBLIC cxx_std_23)
