add_library(
  vma-c STATIC
  include/vma.h
  src/vma.cpp
)

target_include_directories(
  vma-c
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  vma-c
  PUBLIC
    VulkanMemoryAllocator
  PRIVATE
    # To include <vulkan/vulkan.h>
    volk
)

set_target_properties(
  vma-c
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)

if(ENABLE_IPO)
  set_target_properties(vma-c PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
