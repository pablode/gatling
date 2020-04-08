gatling_add_executable(
  gatling
  SOURCES_C
    main.c
  SOURCES_GLSL
    kernel_ray_gen.comp
    kernel_extend.comp
    kernel_shade.comp
)

set_target_properties(
  gatling PROPERTIES
  SHADER_OUTPUT_DIRECTORY "${GATLING_OUTPUT_DIR}/shaders"
)

target_compile_definitions(
  gatling PRIVATE
  GATLING_VERSION_MAJOR=${GATLING_VERSION_MAJOR}
  GATLING_VERSION_MINOR=${GATLING_VERSION_MINOR}
  GATLING_VERSION_PATCH=${GATLING_VERSION_PATCH}
)

target_link_libraries(
  gatling PRIVATE
  cgpu
)
