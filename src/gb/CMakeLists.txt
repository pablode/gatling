add_library(
  gb STATIC
  include/HandleStore.h
  include/LinearDataStore.h
  include/Log.h
  include/SmallVector.h
  src/HandleStore.cpp
  src/LinearDataStore.cpp
  src/Log.cpp
  src/SmallVector.cpp
)

target_include_directories(
  gb
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  gb
  PUBLIC
    quill
)

# Required since library is linked into hdGatling DSO
set_target_properties(gb PROPERTIES POSITION_INDEPENDENT_CODE ON)
