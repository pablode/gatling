
gatling_add_executable(
  gatling
  SOURCES_CXX
    main.cpp
  SOURCES_GLSL
    test.comp
)

target_compile_definitions(
  gatling PRIVATE
  GATLING_VERSION_MAJOR=${GATLING_VERSION_MAJOR}
  GATLING_VERSION_MINOR=${GATLING_VERSION_MINOR}
  GATLING_VERSION_PATCH=${GATLING_VERSION_PATCH}
)

target_include_directories(
  gatling PRIVATE
  ${GATLING_SOURCE_DIR}
)

target_link_libraries(
  gatling PRIVATE
  cgpu
)
