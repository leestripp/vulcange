cmake_minimum_required(VERSION 3.28)
project(vulcanGE VERSION 0.1.0 LANGUAGES CXX)

# ---- Global settings (Plan §2 / §18.1) ----
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VGE_BUILD_TESTS "Build vulcanGE test suite" ON)
option(VGE_BUILD_APP   "Build vulcanGE sandbox application" ON)
set(VGE_SANITIZER "" CACHE STRING "Sanitizer preset: address | undefined | thread")

# ---- Warning policy: zero warnings tolerated (Plan §19) ----
add_library(vge_warnings INTERFACE)
target_compile_options(vge_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall -Wextra -Wpedantic -Werror
        -Wconversion -Wsign-conversion -Wold-style-cast
        -Wno-missing-field-initializers> # Vulkan designated-init idiom
)

# ---- Sanitizer presets (Plan §16.4) ----
if(VGE_SANITIZER)
    message(STATUS "vulcanGE: sanitizer enabled -> ${VGE_SANITIZER}")
    add_compile_options(-fsanitize=${VGE_SANITIZER} -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${VGE_SANITIZER})
endif()

# ---- Third-party: VulkanMemoryAllocator (Plan §16 VMA budgets/tracking) ----
include(FetchContent)
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.3.0
    GIT_SHALLOW    TRUE)
set(VMA_STATIC_VULKAN_FUNCTIONS OFF CACHE BOOL "" FORCE)
set(VMA_DYNAMIC_VULKAN_FUNCTIONS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vma)

add_subdirectory(engine/math)
add_subdirectory(engine/async)
add_subdirectory(engine/platform)
add_subdirectory(engine/gpu)
add_subdirectory(engine/render)

if(VGE_BUILD_APP)
    add_subdirectory(app)
endif()

if(VGE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG        v2.4.12
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(doctest)

    enable_testing()
    add_subdirectory(tests)
endif()
