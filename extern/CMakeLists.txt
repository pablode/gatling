include(FetchContent)

#
# Add dependencies using CMake's FetchContent module.
#

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
