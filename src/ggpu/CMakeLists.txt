set(GGPU_SRCS
  include/denseDataStore.h
  include/linearDataStore.h
  include/resizableBuffer.h
  include/stager.h
  include/syncBuffer.h
  src/denseDataStore.cpp
  src/linearDataStore.cpp
  src/resizableBuffer.cpp
  src/stager.cpp
  src/syncBuffer.cpp
)

function(configure_target TARGET)

  target_link_libraries(${TARGET} PRIVATE gb cgpu)

  target_include_directories(${TARGET}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE src
  )

  # Required since library is linked into hdGatling DSO
  set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

endfunction()

add_library(ggpu STATIC ${GGPU_SRCS})
configure_target(ggpu)

add_executable(ggpu_test ${GGPU_SRCS} src/main.cpp)
target_compile_definitions(ggpu_test PRIVATE TEST_EXECUTABLE)
target_link_libraries(ggpu_test PRIVATE doctest)
configure_target(ggpu_test)
