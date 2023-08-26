add_library(
  cgpu STATIC
  include/cgpu.h
  src/cgpu.cpp
  src/shaderReflection.cpp
  src/shaderReflection.h
)

target_include_directories(
  cgpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  cgpu
  PRIVATE
    gb
    volk
    VulkanMemoryAllocator
    spirv-reflect-static
)

set_target_properties(
  cgpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
