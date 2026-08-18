include(${CMAKE_SOURCE_DIR}/cmake/VgeShaders.cmake)
vge_compile_shaders(VGE_TEST_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gbuffer.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fullscreen.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lighting.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/present.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/meshlet.task
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/meshlet.mesh
)

add_executable(vge_tests
    test_main.cpp
    math_tests.cpp
    async_tests.cpp
    gpu_tests.cpp
    render_tests.cpp
    deferred_tests.cpp
    meshlet_tests.cpp
    ${VGE_TEST_SHADERS}
)
target_link_libraries(vge_tests PRIVATE vge::math vge::async vge::platform vge::gpu vge::render doctest::doctest vge_warnings)
target_compile_definitions(vge_tests PRIVATE VGE_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

# Sanitizer suppression files for third-party DSO noise (GLFW's Wayland backend
# loads libdecor-gtk -> gtk/glib/pango/glycin; Mesa driver thread init). Only the
# active sanitizer's env var is set: doctest_discover_tests flattens PROPERTIES
# lists, so a multi-entry ENVIRONMENT gets misparsed as extra key/value pairs.
set(_vge_san_env "")
if(VGE_SANITIZER STREQUAL "address")
    set(_vge_san_env "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp")
elseif(VGE_SANITIZER STREQUAL "thread")
    set(_vge_san_env "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan.supp")
endif()

doctest_discover_tests(vge_tests
    PROPERTIES ENVIRONMENT "${_vge_san_env}"
)
