add_library(
  gi STATIC
  include/gi.h
  src/gi.cpp
  src/asset_reader.h
  src/asset_reader.cpp
  src/mmap.h
  src/mmap.cpp
  src/stager.h
  src/stager.cpp
  src/texsys.h
  src/texsys.cpp
  src/turbo.h
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
    gml
    imgio
)

# Required since library is linked into hdGatling DSO
set_target_properties(gi PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ENABLE_IPO)
  set_target_properties(gi PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
