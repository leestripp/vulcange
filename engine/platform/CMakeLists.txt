add_library(vge_platform STATIC
    src/window.cpp
)
add_library(vge::platform ALIAS vge_platform)
target_include_directories(vge_platform PUBLIC include)
find_package(Vulkan REQUIRED)
find_package(glfw3 3.4 REQUIRED)
target_link_libraries(vge_platform PUBLIC Vulkan::Headers PRIVATE glfw)
target_compile_features(vge_platform PUBLIC cxx_std_23)
target_link_libraries(vge_platform PRIVATE vge_warnings)
