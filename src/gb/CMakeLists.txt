add_library(
  gb STATIC
  include/handleStore.h
  include/linearDataStore.h
  include/smallVector.h
  src/handleStore.cpp
  src/linearDataStore.cpp
  src/smallVector.cpp
)

target_include_directories(
  gb
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)
