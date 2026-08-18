# vulcanGE::math — header-only (modules-fallback configuration, Plan §18.1)
add_library(vge_math INTERFACE)
add_library(vge::math ALIAS vge_math)

target_include_directories(vge_math INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vge_math INTERFACE vge_warnings)
