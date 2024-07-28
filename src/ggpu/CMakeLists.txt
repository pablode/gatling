add_library(
  ggpu STATIC
  gtl/ggpu/DelayedResourceDestroyer.h
  gtl/ggpu/DenseDataStore.h
  gtl/ggpu/LinearDataStore.h
  gtl/ggpu/ResizableBuffer.h
  gtl/ggpu/Stager.h
  gtl/ggpu/SyncBuffer.h
  impl/DelayedResourceDestroyer.cpp
  impl/DenseDataStore.cpp
  impl/LinearDataStore.cpp
  impl/ResizableBuffer.cpp
  impl/Stager.cpp
  impl/SyncBuffer.cpp
)

target_include_directories(
  ggpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    gtl/ggpu
    impl
)

target_link_libraries(
  ggpu
  PRIVATE
    gb
    cgpu
)

target_compile_options(
  ggpu
  PRIVATE
    ${GTL_COMPILE_OPTIONS}
)

set_target_properties(
  ggpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
