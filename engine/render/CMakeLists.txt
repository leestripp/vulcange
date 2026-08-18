add_library(vge_render STATIC
    src/frame_graph.cpp
    src/bindless_table.cpp
)
add_library(vge::render ALIAS vge_render)
target_include_directories(vge_render PUBLIC include)
target_link_libraries(vge_render PUBLIC vge::gpu PRIVATE vge_warnings)
target_compile_features(vge_render PUBLIC cxx_std_23)
