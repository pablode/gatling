set(GGPU_SRCS
  gtl/ggpu/BumpAllocator.h
  gtl/ggpu/DeleteQueue.h
  gtl/ggpu/DenseDataStore.h
  gtl/ggpu/LinearDataStore.h
  gtl/ggpu/ResizableBuffer.h
  gtl/ggpu/Stager.h
  gtl/ggpu/SyncBuffer.h
  impl/BumpAllocator.cpp
  impl/DeleteQueue.cpp
  impl/DenseDataStore.cpp
  impl/LinearDataStore.cpp
  impl/ResizableBuffer.cpp
  impl/Stager.cpp
  impl/SyncBuffer.cpp
)

function(configure_target TARGET)
  target_include_directories(
    ${TARGET}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
      gtl/ggpu
      impl
  )

  target_link_libraries(
    ${TARGET}
    PRIVATE
      gb
      cgpu
  )
endfunction()

add_library(ggpu STATIC ${GGPU_SRCS})
set_target_properties(
  ggpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
configure_target(ggpu)

add_executable(ggpu_test ${GGPU_SRCS} impl/main.cpp)
target_compile_definitions(ggpu_test PRIVATE GGPU_TEST_EXECUTABLE=1)
target_link_libraries(ggpu_test PRIVATE gt doctest)
configure_target(ggpu_test)
