add_library(
  gi STATIC
  include/gi.h
  src/gi.cpp
  src/bvh.cpp
  src/bvh.h
  src/bvh_collapse.cpp
  src/bvh_collapse.h
  src/bvh_compress.cpp
  src/bvh_compress.h
)

set_target_properties(
  gi
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
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

# Required since library is linked into hdGatling DSO
set_target_properties(gi PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ENABLE_IPO)
  set_target_properties(gi PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
