add_executable(
  gatling
  img.c
  img.h
  main.c
  mmap.c
  mmap.h
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
  ${C_MATH_LIBRARY}
  png
)

add_shader_library(
  gatling-shaders
  shaders/main.comp
  INCLUDES
    shaders/bvh.glsl
    shaders/common.glsl
    shaders/extensions.glsl
)

set_target_properties(
  gatling-shaders PROPERTIES
  SHADER_OUTPUT_DIRECTORY "${GATLING_OUTPUT_DIR}/shaders"
)

add_dependencies(
  gatling
  gatling-shaders
)
