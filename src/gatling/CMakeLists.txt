add_executable(
  gatling
  main.c
)

target_compile_definitions(
  gatling PRIVATE
  GATLING_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  GATLING_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  GATLING_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

target_link_libraries(
  gatling PRIVATE
  cgpu
)

add_shader_library(
  gatling-shaders
  kernel_ray_gen.comp
  kernel_extend.comp
  kernel_shade.comp
  INCLUDES
    common.glsl
)

set_target_properties(
  gatling-shaders PROPERTIES
  SHADER_OUTPUT_DIRECTORY "${GATLING_OUTPUT_DIR}/shaders"
)

add_dependencies(
  gatling
  gatling-shaders
)
