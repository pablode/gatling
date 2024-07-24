add_library(
  gt STATIC
  gtl/gt/LogFlushListener.h
  impl/LogFlushListener.cpp
)

target_include_directories(
  gt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    gtl/gt
    _impl
)

target_link_libraries(
  gt
  PUBLIC
    doctest
  PRIVATE
    gb
)
