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

if(WIN32)
  target_compile_definitions(
    gi
    PRIVATE
      NOMINMAX
      _USE_MATH_DEFINES
  )
endif()

target_link_libraries(
  gi
  PRIVATE
    ${C_MATH_LIBRARY}
    gml
)
