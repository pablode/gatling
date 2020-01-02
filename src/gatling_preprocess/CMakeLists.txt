add_executable(
  gatling_preprocess
  bvh.c
  bvh.h
  gp.h
  main.c
  math.c
  math.h
)

target_compile_definitions(
  gatling_preprocess PRIVATE
  GATLING_VERSION_MAJOR=${GATLING_VERSION_MAJOR}
  GATLING_VERSION_MINOR=${GATLING_VERSION_MINOR}
  GATLING_VERSION_PATCH=${GATLING_VERSION_PATCH}
  _USE_MATH_DEFINES
)

target_link_libraries(
  gatling_preprocess PRIVATE
  assimp
  m
)
