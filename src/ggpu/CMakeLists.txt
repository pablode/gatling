add_library(
  ggpu STATIC
  include/DenseDataStore.h
  include/LinearDataStore.h
  include/ResizableBuffer.h
  include/Stager.h
  include/SyncBuffer.h
  src/DenseDataStore.cpp
  src/LinearDataStore.cpp
  src/ResizableBuffer.cpp
  src/Stager.cpp
  src/SyncBuffer.cpp
)

target_include_directories(
  ggpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  ggpu
  PRIVATE
    gb
    cgpu
)

set_target_properties(
  ggpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
