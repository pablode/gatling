include(FetchContent)

set(BUILD_SHARED_LIBS OFF)

# Make option() honor normal variables.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

#
# volk
#
add_subdirectory(volk)

#
# MaterialX
#
if(NOT ${GATLING_BUILD_HDGATLING})

set(MATERIALX_BUILD_GEN_GLSL OFF)
set(MATERIALX_BUILD_GEN_OSL OFF)
set(MATERIALX_BUILD_RENDER OFF)
set(MATERIALX_BUILD_TESTS OFF)
set(MATERIALX_INSTALL_PYTHON OFF)
set(MATERIALX_TEST_RENDER OFF)
add_subdirectory(MaterialX)

endif()

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MaterialX/source/MaterialXGenMdl/mdl/materialx"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources/mdl"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MaterialX/libraries/"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources/materialx"
  COMPONENT hdGatling
  FILES_MATCHING
  PATTERN "*.mtlx"
)

#
# VulkanMemoryAllocator
#
set(VMA_STATIC_VULKAN_FUNCTIONS OFF)
set(VMA_DYNAMIC_VULKAN_FUNCTIONS OFF)
add_subdirectory(VulkanMemoryAllocator)

#
# shaderc dependencies. The revisions are taken from the repo/DEPS.
#
if(${GATLING_USE_GLSLANG})

# SPIRV-Headers
set(SPIRV_HEADERS_SKIP_EXAMPLES ON)
set(SPIRV_HEADERS_SKIP_INSTALL ON)
add_subdirectory(SPIRV-Headers)

# SPIRV-Tools
set(SPIRV_SKIP_EXECUTABLES ON)
add_subdirectory(SPIRV-Tools)

# glslang
set(ENABLE_SPRVREMAPPER OFF)
set(ENABLE_CTEST OFF)
add_subdirectory(glslang)

# shaderc
set(SHADERC_SKIP_EXAMPLES ON)
set(SHADERC_SKIP_TESTS ON)
set(SHADERC_SKIP_INSTALL ON)
set(SHADERC_ENABLE_SHARED_CRT ON)
add_subdirectory(shaderc)

endif()

#
# Embree
#

if(${GATLING_USE_EMBREE})
  set(DEFAULT_CMAKE_CONFIGURATION_TYPES_SET ON)
  set(EMBREE_STATIC_LIB ON)
  set(EMBREE_TUTORIALS OFF)
  set(EMBREE_ISPC_SUPPORT OFF)
  set(EMBREE_GEOMETRY_QUAD OFF)
  set(EMBREE_GEOMETRY_CURVE OFF)
  set(EMBREE_GEOMETRY_SUBDIVISION OFF)
  set(EMBREE_GEOMETRY_USER OFF)
  set(EMBREE_GEOMETRY_INSTANCE OFF)
  set(EMBREE_GEOMETRY_GRID OFF)
  set(EMBREE_GEOMETRY_POINT OFF)
  set(EMBREE_RAY_PACKETS OFF)
  set(EMBREE_TASKING_SYSTEM "" CACHE STRING "Don't use TBB." FORCE)
  add_subdirectory(embree)
endif()

#
# zlib-ng
#
set(ZLIB_COMPAT ON)
set(ZLIB_ENABLE_TESTS OFF)
add_subdirectory(zlib-ng)
target_include_directories(zlib PUBLIC zlib-ng $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

#
# libspng
#
add_library(spng STATIC libspng/spng/spng.c)
target_include_directories(spng PUBLIC libspng/spng)
target_link_libraries(spng PRIVATE zlib)
target_compile_definitions(spng PUBLIC SPNG_STATIC)

#
# libjpeg-turbo
#
set(CMAKE_INSTALL_DOCDIR "${CMAKE_CURRENT_BINARY_DIR}/doc" CACHE PATH "" FORCE)
add_subdirectory(libjpeg-turbo)
target_include_directories(turbojpeg-static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo>)

#
# Imath
#
add_subdirectory(Imath)

#
# OpenEXR
#
set(OPENEXR_INSTALL OFF)
set(OPENEXR_INSTALL_TOOLS OFF)
set(OPENEXR_INSTALL_EXAMPLES OFF)
add_subdirectory(OpenEXR)

#
# SPIRV-Reflect
#
set(SPIRV_REFLECT_EXECUTABLE OFF)
set(SPIRV_REFLECT_EXAMPLES OFF)
set(SPIRV_REFLECT_STATIC_LIB ON)
add_subdirectory(SPIRV-Reflect)
