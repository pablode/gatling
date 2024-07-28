add_library(
  cgpu STATIC
  gtl/cgpu/Cgpu.h
  impl/Cgpu.cpp
  impl/ShaderReflection.cpp
  impl/ShaderReflection.h
)

target_include_directories(
  cgpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    gtl/cgpu
    impl
)

target_link_libraries(
  cgpu
  PRIVATE
    gb
    volk
    VulkanMemoryAllocator
    spirv-reflect-static
)

target_compile_options(
  cgpu
  PRIVATE
    ${GTL_COMPILE_OPTIONS}
)

set_target_properties(
  cgpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
