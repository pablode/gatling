
add_executable(
  gatling_preprocess
  main.c
)

target_compile_definitions(
  gatling_preprocess PRIVATE
  GATLING_VERSION_MAJOR=${GATLING_VERSION_MAJOR}
  GATLING_VERSION_MINOR=${GATLING_VERSION_MINOR}
  GATLING_VERSION_PATCH=${GATLING_VERSION_PATCH}
)

target_include_directories(
  gatling_preprocess PRIVATE
  ${GATLING_SOURCE_DIR}
)

target_link_libraries(
  gatling_preprocess PRIVATE
  assimp
)
