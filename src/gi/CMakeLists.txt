add_library(
  gi
  include/gi.h
  src/gi.c
  src/bvh.c
  src/bvh.h
  src/bvh_collapse.c
  src/bvh_collapse.h
  src/bvh_compress.c
  src/bvh_compress.h
)

target_include_directories(
  gi
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    src
)

target_compile_definitions(
  gi
  PRIVATE
    GATLING_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GATLING_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GATLING_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

target_link_libraries(
  gi
  PRIVATE
    cgpu
    shadergen
    ${C_MATH_LIBRARY}
    gml
)

if(ENABLE_IPO)
  set_target_properties(gi PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
