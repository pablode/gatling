add_library(
  gb STATIC
  include/SmallVector.h
  src/SmallVector.cpp
)

target_include_directories(
  gb
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

if(ENABLE_IPO)
  set_target_properties(gb PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
