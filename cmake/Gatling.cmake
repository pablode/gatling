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

# Define gatling_add_executable function, which takes multiple arguments:
# SOURCES_C, for all application C source files,
# SOURCES_GLSL, for all shaders which should be compiled
# It also respects the SHADER_OUTPUT_DIRECTORY target property.
function(gatling_add_executable target)

  # Read arguments, create target and set SHADER_OUTPUT_DIRECTORY default value.
  cmake_parse_arguments(TARGET_EXE "" "" "SOURCES_C;SOURCES_GLSL" ${ARGN})

  add_executable(${target} ${TARGET_EXE_SOURCES_C} ${TARGET_EXE_UNPARSED_ARGS})

  get_target_property(TARGET_RUNTIME_OUTPUT_DIRECTORY ${target} RUNTIME_OUTPUT_DIRECTORY)
  set_target_properties(${target} PROPERTIES SHADER_OUTPUT_DIRECTORY ${TARGET_RUNTIME_OUTPUT_DIRECTORY})

  # Get input and output names and paths of soon-to-be-compiled shaders.
  list(APPEND TARGET_SHADERS_INPUT_FILE_PATHS "")
  list(APPEND TARGET_SHADERS_INPUT_FILE_NAMES "")
  list(APPEND TARGET_SHADERS_OUTPUT_FILE_NAMES "")
  list(APPEND TARGET_SHADERS_OUTPUT_FILE_PATHS "")
  foreach(shader ${TARGET_EXE_SOURCES_GLSL})
    get_filename_component(shader_path_abs ${shader} ABSOLUTE)
    get_filename_component(shader_name ${shader} NAME)
    set(spv_file_name "${shader_name}.spv")
    set(spv_file_path "${CMAKE_CURRENT_BINARY_DIR}/${spv_file_name}")
    list(APPEND TARGET_SHADERS_INPUT_FILE_PATHS ${shader_path_abs})
    list(APPEND TARGET_SHADERS_INPUT_FILE_NAMES ${shader_name})
    list(APPEND TARGET_SHADERS_OUTPUT_FILE_NAMES ${spv_file_name})
    list(APPEND TARGET_SHADERS_OUTPUT_FILE_PATHS ${spv_file_path})
  endforeach()

  # Create an intermediate target for compilation dependencies.
  set(target_shaders "${target}-shaders")
  add_custom_target(${target_shaders} DEPENDS ${TARGET_SHADERS_OUTPUT_FILE_PATHS})
  add_dependencies(${target} ${target_shaders})

  # Make sure the path to the shader output directory exists.
  add_custom_command(
     TARGET ${target_shaders} PRE_BUILD
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
    )

    # Always copy cached compiled shaders to output dir. Note that although the
    # function name is the same as above, the commands do very different things.
    add_custom_command(
      TARGET ${target_shaders} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${output_path} "$<TARGET_PROPERTY:${target},SHADER_OUTPUT_DIRECTORY>/${output_name}"
      VERBATIM
    )
  endforeach()

endfunction()
