include(FetchContent)

# Make option() honor normal variables.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

#
# volk
#
add_subdirectory(volk)

#
# MaterialX
#
set(MATERIALX_BUILD_GEN_GLSL OFF)
set(MATERIALX_BUILD_GEN_OSL OFF)
set(MATERIALX_BUILD_RENDER OFF)
set(MATERIALX_BUILD_TESTS OFF)
set(MATERIALX_INSTALL_PYTHON OFF)
set(MATERIALX_TEST_RENDER OFF)
add_subdirectory(MaterialX)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MaterialX/source/MaterialXGenMdl/mdl/materialx"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources/mtlxmdl"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MaterialX/libraries/"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources/mtlxlib"
  COMPONENT hdGatling
  FILES_MATCHING
  PATTERN "*.mdl"
  PATTERN "*.mtlx"
)

#
# shaderc dependencies. The revisions are taken from the repo/DEPS.
#
if(NOT ${GATLING_USE_DXC})

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
