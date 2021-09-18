include(CMakeParseArguments)

function(add_shader_library target)

  # Read shader files and includes
  cmake_parse_arguments("TARGET" "" "" "INCLUDES" "${ARGN}")

  # "Namespace" header files so we know that they are generated
  set(header_dir "${CMAKE_CURRENT_BINARY_DIR}/SPV")

  # Provide a compilation command for each shader file
  set(header_paths "")

  foreach(shader ${TARGET_UNPARSED_ARGUMENTS})
    get_filename_component(shader_path ${shader} ABSOLUTE)
    get_filename_component(shader_name ${shader} NAME)
    get_filename_component(shader_name_we ${shader} NAME_WE)
    get_filename_component(shader_ext ${shader} EXT)
    string(SUBSTRING ${shader_ext} 1 -1 shader_ext)

    set(header_path "${header_dir}/${shader_name}.spv.h")

    add_custom_command(
      OUTPUT ${header_path}
      MAIN_DEPENDENCY ${shader_path}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Compiling GLSL shader ${shader_name}"
      COMMAND ${CMAKE_COMMAND} -E make_directory ${header_dir}
      COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -D -e CSMain --target-env vulkan1.1 --vn "SPV_${shader_name_we}_${shader_ext}" -o ${header_path} --quiet ${shader_path}
      DEPENDS ${TARGET_INCLUDES}
    )

    list(APPEND header_paths ${header_path})

  endforeach()

  # Create a library which depends on the generated header files
  add_library(${target} INTERFACE)
  target_sources(${target} INTERFACE ${header_paths})
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

endfunction()
