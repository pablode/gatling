if (APPLE)
  set(CGPU_IMPL_FILE_NAME "impl/CgpuMtl.cpp")
else()
  set(CGPU_IMPL_FILE_NAME "impl/CgpuVk.cpp")
endif()

add_library(
  cgpu STATIC
  gtl/cgpu/Cgpu.h
  ${CGPU_IMPL_FILE_NAME}
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

if (APPLE)
  set(
    CGPU_IMPL_LIBRARIES
      metal-cpp
      spirv-cross-c
      "-framework Metal"
      "-framework Foundation"
  )
else()
  set(
    CGPU_IMPL_LIBRARIES
      volk
      VulkanMemoryAllocator
  )
endif()

target_compile_definitions(
  cgpu
  PRIVATE
    CGPU_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    CGPU_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    CGPU_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

target_link_libraries(
  cgpu
  PRIVATE
    gb
    spirv-reflect-static
    ${CGPU_IMPL_LIBRARIES}
)

set_target_properties(
  cgpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
