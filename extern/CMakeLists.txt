#
# Define policies for old CMake project versions. These defaults only affect
# the current folder and its subfolders. Policy values have to be evaluated
# per-case.
#

# Make option() honor normal variables.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

#
# Add dependency targets using CMake's FetchContent module.
#
include(FetchContent)

# assimp
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_INSTALL OFF)
FetchContent_Declare(
  assimp
  GIT_REPOSITORY https://github.com/assimp/assimp.git
  GIT_TAG v5.0.1
)
FetchContent_MakeAvailable(assimp)

# volk
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG 1.2.140
)
FetchContent_MakeAvailable(volk)
