include(FetchContent)

# Make option() honor normal variables.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# volk
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG 1.2.140
)
FetchContent_MakeAvailable(volk)

#
# shaderc dependencies. The revisions are taken from the repo/DEPS.
#

# SPIRV-Headers
set(SPIRV_HEADERS_SKIP_EXAMPLES ON)
set(SPIRV_HEADERS_SKIP_INSTALL ON)
FetchContent_Declare(
  spirv-headers
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers
  GIT_TAG 449bc986ba6f4c5e10e32828783f9daef2a77644
)
FetchContent_MakeAvailable(spirv-headers)

# SPIRV-Tools
set(SPIRV_SKIP_EXECUTABLES ON)
FetchContent_Declare(
  spirv-tools
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools
  GIT_TAG 1fbed83c8aab8517d821fcb4164c08567951938f
)
FetchContent_MakeAvailable(spirv-tools)

# glslang
set(ENABLE_SPRVREMAPPER OFF)
set(ENABLE_CTEST OFF)
FetchContent_Declare(
  glslang
  GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
  GIT_TAG a4599ef7561abed83d45bab4c7492daeceef92a5
)
FetchContent_MakeAvailable(glslang)

# shaderc
set(SHADERC_SKIP_EXAMPLES ON)
set(SHADERC_SKIP_TESTS ON)
set(SHADERC_SKIP_INSTALL ON)
set(SHADERC_ENABLE_SHARED_CRT ON)
FetchContent_Declare(
  shaderc
  GIT_REPOSITORY https://github.com/google/shaderc
  GIT_TAG v2021.2
)
FetchContent_MakeAvailable(shaderc)
