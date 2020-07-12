add_executable(
  gp
  bvh.c
  bvh.h
  gp.h
  main.c
  math.c
  math.h
)

target_compile_definitions(
  gp PRIVATE
  GATLING_VERSION_MAJOR=${GATLING_VERSION_MAJOR}
  GATLING_VERSION_MINOR=${GATLING_VERSION_MINOR}
  GATLING_VERSION_PATCH=${GATLING_VERSION_PATCH}
  _USE_MATH_DEFINES
)

target_link_libraries(
  gp PRIVATE
  assimp
  m
)
