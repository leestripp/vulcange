# The demo renders with the same shaders as the test suite; the tests own
# their compilation (build order via add_dependencies).
add_executable(vge_sandbox main.cpp)
target_link_libraries(vge_sandbox PRIVATE vge::math vge::platform vge::gpu vge::render vge_warnings)
target_compile_definitions(vge_sandbox PRIVATE VGE_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
add_dependencies(vge_sandbox vge_tests)
