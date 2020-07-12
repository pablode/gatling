include(CMakeParseArguments)

# Try to find a GLSL compiler.
if (NOT CMAKE_GLSL_COMPILER)
  find_program(CMAKE_GLSL_COMPILER glslc)
endif()

if (CMAKE_GLSL_COMPILER)
  message(STATUS "Using GLSL compiler: ${CMAKE_GLSL_COMPILER}")
else()
  message(FATAL_ERROR "No GLSL compiler found.")
endif()

# Define SHADER_OUTPUT_DIRECTORY target property.
define_property(
  TARGET
  PROPERTY SHADER_OUTPUT_DIRECTORY
  BRIEF_DOCS "The directory where SPIR-V files are written to."
  FULL_DOCS "The directory where SPIR-V files are written to. Has default value of RUNTIME_OUTPUT_DIRECTORY."
)

# The add_shader_library function produces a target which, when built, invokes the
# GLSL compiler on the given shader files. It provides an INCLUDES section where
# files can be defined which are included by the shaders. This is necessary since
# we want to recompile a shader each time included files change. Unfortunately, for
# now, we can only recompile on a per-target basis. The target property
# SHADER_OUTPUT_DIRECTORY can be set to define the compilation output directory.
# It's initialized with the value of CMAKE_RUNTIME_OUTPUT_DIRECTORY.
function(add_shader_library target)

  # Read input args, extract shader file paths and names.
  cmake_parse_arguments("TARGET" "" "" "INCLUDES" ${ARGN})

  list(APPEND TARGET_SHADERS_INPUT_FILE_PATHS "")
  list(APPEND TARGET_SHADERS_INPUT_FILE_NAMES "")
  list(APPEND TARGET_SHADERS_OUTPUT_FILE_NAMES "")
  list(APPEND TARGET_SHADERS_OUTPUT_FILE_PATHS "")
  foreach(shader ${TARGET_UNPARSED_ARGUMENTS})
    get_filename_component(shader_path_abs ${shader} ABSOLUTE)
    get_filename_component(shader_name ${shader} NAME)
    set(spv_file_name "${shader_name}.spv")
    set(spv_file_path "${CMAKE_CURRENT_BINARY_DIR}/${spv_file_name}")
    list(APPEND TARGET_SHADERS_INPUT_FILE_PATHS ${shader_path_abs})
    list(APPEND TARGET_SHADERS_INPUT_FILE_NAMES ${shader_name})
    list(APPEND TARGET_SHADERS_OUTPUT_FILE_NAMES ${spv_file_name})
    list(APPEND TARGET_SHADERS_OUTPUT_FILE_PATHS ${spv_file_path})
  endforeach()

  # Create a custom target, set initial value of output directory property.
  add_custom_target(${target} DEPENDS ${TARGET_SHADERS_OUTPUT_FILE_PATHS})
  set_target_properties(${target} PROPERTIES SHADER_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # Make sure the path to the shader output directory exists when compiling.
  add_custom_command(
     TARGET ${target} PRE_BUILD
     COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_PROPERTY:${target},SHADER_OUTPUT_DIRECTORY>
     VERBATIM
  )

  # Add custom commands to compile the shaders at build time.
  list(LENGTH TARGET_SHADERS_INPUT_FILE_PATHS SHADER_FILE_COUNT)
  math(EXPR SHADER_FILE_COUNT "${SHADER_FILE_COUNT}-1")
  foreach(i RANGE ${SHADER_FILE_COUNT})
    list(GET TARGET_SHADERS_INPUT_FILE_PATHS ${i} input_path)
    list(GET TARGET_SHADERS_INPUT_FILE_NAMES ${i} input_name)
    list(GET TARGET_SHADERS_OUTPUT_FILE_NAMES ${i} output_name)
    list(GET TARGET_SHADERS_OUTPUT_FILE_PATHS ${i} output_path)

    # Unfortunately, we have to use a workaround because of this issue:
    # https://gitlab.kitware.com/cmake/cmake/issues/12877
    # Instead of outputting directly to the specified directory, we write
    # to a temporary location and copy the files on every build invocation.
    add_custom_command(
      OUTPUT ${output_path}
      MAIN_DEPENDENCY ${input_path}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Compiling GLSL shader ${input_name}"
      COMMAND ${CMAKE_GLSL_COMPILER} -o ${output_path} -c ${input_path}
      VERBATIM
      # Rebuild if an included file changes.
      DEPENDS ${TARGET_INCLUDES}
    )

    # Always copy cached compiled shaders to output dir. Note that although the
    # function name is the same as above, the commands do very different things.
    add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${output_path} "$<TARGET_PROPERTY:${target},SHADER_OUTPUT_DIRECTORY>/${output_name}"
      VERBATIM
    )
  endforeach()

endfunction()
