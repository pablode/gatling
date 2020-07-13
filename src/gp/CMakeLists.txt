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
  GATLING_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  GATLING_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  GATLING_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  _USE_MATH_DEFINES
)

if (NOT MSVC)
  set(MATH_LIB "m")
else()
  set(MATH_LIB "")
endif()

target_link_libraries(
  gp PRIVATE
  assimp
  ${MATH_LIB}
)
