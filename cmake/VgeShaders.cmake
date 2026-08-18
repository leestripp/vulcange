# vge_compile_shaders(<out_var> <shader files...>)
# Compiles GLSL -> SPIR-V (Vulkan 1.3 target) into ${CMAKE_BINARY_DIR}/shaders
# via glslangValidator. Runtime hot-reload via shaderc comes later (Plan §12).
find_program(VGE_GLSLANG_VALIDATOR glslangValidator REQUIRED)

function(vge_compile_shaders out_var)
    set(spv_files)
    foreach(src IN LISTS ARGN)
        get_filename_component(name "${src}" NAME)
        set(out "${CMAKE_BINARY_DIR}/shaders/${name}.spv")
        add_custom_command(
            OUTPUT "${out}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
            COMMAND "${VGE_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 "${src}" -o "${out}"
            DEPENDS "${src}"
            VERBATIM)
        list(APPEND spv_files "${out}")
    endforeach()
    set(${out_var} ${spv_files} PARENT_SCOPE)
endfunction()
