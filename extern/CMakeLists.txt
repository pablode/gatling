include(FetchContent)

#
# Add dependencies using CMake's FetchContent module.
#

# zlib
set(SKIP_INSTALL_ALL ON)
# The AMD64 option causes an MSVC linker error and the ASM build has problems:
# https://github.com/madler/zlib/issues/41#issuecomment-125848075
set(AMD64 OFF)
set(ASM686 OFF)
FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.11
)
FetchContent_MakeAvailable(zlib)
unset(SKIP_INSTALL_ALL)
unset(ASM686)
unset(AMD64)

# assimp
get_target_property(ZLIB_SOURCE_DIR zlib SOURCE_DIR)
get_target_property(ZLIB_BINARY_DIR zlib BINARY_DIR)
set(ZLIB_INCLUDE_DIRS ${ZLIB_BINARY_DIR} ${ZLIB_SOURCE_DIR})
set(ZLIB_INCLUDE_DIR ${ZLIB_BINARY_DIR} ${ZLIB_SOURCE_DIR})
set(ZLIB_LIBRARY $<TARGET_FILE:zlib>)
set(ZLIB_LIBRARIES $<TARGET_FILE:zlib>)
set(ZLIB_FOUND TRUE)

set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
set(ASSIMP_BUILD_MINIZIP ON)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_BUILD_ZLIB ON)
set(ASSIMP_INSTALL OFF)
FetchContent_Declare(
  assimp
  GIT_REPOSITORY https://github.com/assimp/assimp.git
  GIT_TAG v5.0.1
)
FetchContent_MakeAvailable(assimp)
add_dependencies(assimp zlib)

# volk
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG 1.2.140
)
FetchContent_MakeAvailable(volk)
