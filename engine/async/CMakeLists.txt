# vulcanGE::async — lock-free work-stealing task system (Plan §4.1)
add_library(vge_async STATIC
    src/task_system.cpp
)
add_library(vge::async ALIAS vge_async)

target_include_directories(vge_async PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vge_async PUBLIC vge_warnings)
find_package(Threads REQUIRED)
target_link_libraries(vge_async PUBLIC Threads::Threads)
