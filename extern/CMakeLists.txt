include(FetchContent)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Make option() honor normal variables.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# target_link_libraries() allows use with targets in other directories
cmake_policy(SET CMP0079 NEW)

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

endif()

#
# Vulkan-Headers
#
add_subdirectory(Vulkan-Headers)

#
# volk
#
set(VOLK_PULL_IN_VULKAN OFF)
add_subdirectory(volk)
# Volk tries to link against the Vulkan::Vulkan target provided by the SDK. We
# use the Vulkan-Headers target (from above submodule with the same name) instead.
target_link_libraries(volk PUBLIC Vulkan-Headers)

#
# VulkanMemoryAllocator
#
set(VMA_STATIC_VULKAN_FUNCTIONS OFF)
set(VMA_DYNAMIC_VULKAN_FUNCTIONS OFF)
add_subdirectory(VulkanMemoryAllocator)
target_link_libraries(VulkanMemoryAllocator PUBLIC $<BUILD_INTERFACE:Vulkan-Headers>)

# SPIRV-Headers (SPIRV-Tools dependency)
set(SPIRV_SKIP_TESTS ON)
set(SPIRV_HEADERS_SKIP_EXAMPLES ON)
set(SPIRV_HEADERS_SKIP_INSTALL ON)
add_subdirectory(SPIRV-Headers)

# SPIRV-Tools (glslang dependency)
set(SPIRV_SKIP_EXECUTABLES ON)
set(SPIRV_WERROR OFF)
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
add_subdirectory(openexr)

#
# stb
#
add_library(stb INTERFACE)
set_target_properties(stb PROPERTIES LINKER_LANGUAGE C)
target_include_directories(stb INTERFACE stb)

#
# SPIRV-Reflect
#
set(SPIRV_REFLECT_EXECUTABLE OFF)
set(SPIRV_REFLECT_EXAMPLES OFF)
set(SPIRV_REFLECT_STATIC_LIB ON)
add_subdirectory(SPIRV-Reflect)
target_compile_definitions(spirv-reflect-static PRIVATE SPIRV_REFLECT_USE_SYSTEM_SPIRV_H)
target_link_libraries(spirv-reflect-static PRIVATE SPIRV-Headers)

#
# GLM
#
add_subdirectory(glm)

#
# efsw
#
add_subdirectory(efsw)

#
# quill
#
add_subdirectory(quill)

#
# doctest
#
set(DOCTEST_NO_INSTALL ON)
add_subdirectory(doctest)
set(DOCTEST_SCRIPT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/doctest/scripts/cmake/doctest.cmake" PARENT_SCOPE)
