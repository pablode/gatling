gatling_add_executable(
  gatling
  SOURCES_C
    main.c
  SOURCES_GLSL
    prim_ray_gen.comp
    trace_ray.comp
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
