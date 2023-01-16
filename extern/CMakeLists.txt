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

# SPIRV-Headers (SPIRV-Tools dependency)
set(SPIRV_HEADERS_SKIP_EXAMPLES ON)
set(SPIRV_HEADERS_SKIP_INSTALL ON)
add_subdirectory(SPIRV-Headers)

# SPIRV-Tools (glslang dependency)
set(SPIRV_SKIP_EXECUTABLES ON)
add_subdirectory(SPIRV-Tools)

#
# glslang
#
set(SKIP_GLSLANG_INSTALL ON)
set(ENABLE_GLSLANG_BINARIES OFF)
set(ENABLE_SPRVREMAPPER OFF)
set(ENABLE_CTEST OFF)
set(ENABLE_OPT) # requires SPIRV-Tools-opt target
add_subdirectory(glslang)

#
# zlib-ng
#
set(ZLIB_COMPAT ON)
set(ZLIB_ENABLE_TESTS OFF)
add_subdirectory(zlib-ng)
target_include_directories(zlib PUBLIC zlib-ng $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
add_library(ZLIB::ZLIB ALIAS zlib) # Needed for OpenEXR

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
