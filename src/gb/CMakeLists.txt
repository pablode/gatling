add_library(
  gb STATIC
  include/smallVector.h
  src/smallVector.cpp
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
