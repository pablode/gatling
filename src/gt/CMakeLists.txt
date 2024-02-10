add_library(
  gt STATIC
  include/LogFlushListener.h
  src/LogFlushListener.cpp
)

target_include_directories(
  gt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  gt
  PUBLIC
    doctest
  PRIVATE
    gb
)
