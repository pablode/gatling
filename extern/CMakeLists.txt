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
get_target_property(ZLIB_SOURCE_DIR zlibstatic SOURCE_DIR)
get_target_property(ZLIB_BINARY_DIR zlibstatic BINARY_DIR)
set(ZLIB_INCLUDE_DIRS ${ZLIB_BINARY_DIR} ${ZLIB_SOURCE_DIR})
set(ZLIB_INCLUDE_DIR ${ZLIB_BINARY_DIR} ${ZLIB_SOURCE_DIR})
set(ZLIB_LIBRARY $<TARGET_FILE:zlibstatic>)
set(ZLIB_LIBRARIES $<TARGET_FILE:zlibstatic>)
set(ZLIB_FOUND TRUE)

# assimp
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
add_dependencies(assimp zlibstatic)

# libpng
set(PNG_BUILD_ZLIB ON)
set(PNG_EXECUTABLES OFF)
set(PNG_SHARED OFF)
set(PNG_STATIC ON)
set(PNG_TESTS OFF)
FetchContent_Declare(
  libpng
  GIT_REPOSITORY https://github.com/glennrp/libpng.git
  GIT_TAG v1.6.35
)
FetchContent_MakeAvailable(libpng)
add_dependencies(png_static zlib)
get_target_property(PNG_BINARY_DIR png_static BINARY_DIR)
get_target_property(PNG_SOURCE_DIR png_static SOURCE_DIR)
target_include_directories(png_static PUBLIC $<BUILD_INTERFACE:${PNG_BINARY_DIR}>)
target_include_directories(png_static PUBLIC $<BUILD_INTERFACE:${PNG_SOURCE_DIR}>)

# volk
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG 1.2.140
)
FetchContent_MakeAvailable(volk)
