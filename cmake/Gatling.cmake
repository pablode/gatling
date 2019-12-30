include(CMakeParseArguments)

function(gatling_add_executable target)

  cmake_parse_arguments(
    EXE
    "NON_DEFAULT"
    ""
    "SOURCES_C;SOURCES_GLSL" ${ARGN}
  )

  set(target_output_dir "${GATLING_OUTPUT_DIR}/${target}")
  file(MAKE_DIRECTORY ${target_output_dir})
  set(target_output_shader_dir "${target_output_dir}/shaders")
  file(MAKE_DIRECTORY ${target_output_shader_dir})

  add_executable(${target}
    ${EXE_SOURCES_C} ${EXE_UNPARSED_ARGS})

  set_target_properties(
    gatling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${target_output_dir})

  foreach(shader ${EXE_SOURCES_GLSL})
    get_filename_component(shader_path_abs ${shader} ABSOLUTE)
    get_filename_component(shader_name ${shader} NAME)
    set(output_file "${target_output_shader_dir}/${shader_name}.spv")
    get_filename_component(output_file ${output_file} ABSOLUTE)
    compile_glsl_using_glslc(${shader_path_abs} ${output_file})
  endforeach()

endfunction()

function(compile_glsl_using_glslc input_file output_file)

  # TODO: properly find compiler
  set(CMAKE_GLSL_COMPILER "glslc")

  add_custom_command(
    TARGET ${target}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Compiling SPIR-V shader ${shader}"
    COMMAND ${CMAKE_GLSL_COMPILER}
      -o ${output_file} -c ${input_file})

endfunction()
