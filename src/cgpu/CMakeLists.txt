if (APPLE)
  set(CGPU_SOURCE_FILE_NAME "impl/Cgpu_metal.cpp")
else()
  # TODO: rename to Cgpu_vulkan.cpp
  set(CGPU_SOURCE_FILE_NAME "impl/Cgpu.cpp")
endif()

add_library(
  cgpu STATIC
  gtl/cgpu/Cgpu.h
  ${CGPU_SOURCE_FILE_NAME}
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

target_link_libraries(cgpu PRIVATE gb spirv-reflect-static)

if (APPLE)
  target_link_libraries(
    cgpu
    PRIVATE
      metal-cpp
      spirv-cross-c
      "-framework Metal"
      "-framework MetalKit"
      "-framework Foundation"
  )
else()
  target_link_libraries(cgpu PRIVATE volk VulkanMemoryAllocator)
endif()

set_target_properties(
  cgpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
