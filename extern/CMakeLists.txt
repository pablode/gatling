include(FetchContent)

#
# Add dependencies using CMake's FetchContent module.
#

set(SKIP_INSTALL_ALL ON)
set(BUILD_SHARED_LIBS OFF)

# zlib
# The AMD64 option causes an MSVC linker error and the ASM build has problems:
# https://github.com/madler/zlib/issues/41#issuecomment-125848075
set(AMD64 OFF CACHE INTERNAL "" FORCE)
set(ASM686 OFF CACHE INTERNAL "" FORCE)
FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.11
)
FetchContent_MakeAvailable(zlib)
unset(ASM686)
unset(AMD64)
get_target_property(ZLIB_SOURCE_DIR zlibstatic SOURCE_DIR)
get_target_property(ZLIB_BINARY_DIR zlibstatic BINARY_DIR)
set(ZLIB_INCLUDE_DIRS ${ZLIB_BINARY_DIR} ${ZLIB_SOURCE_DIR})
set(ZLIB_INCLUDE_DIR ${ZLIB_BINARY_DIR} ${ZLIB_SOURCE_DIR})
set(ZLIB_LIBRARY $<TARGET_FILE:zlibstatic>)
set(ZLIB_LIBRARIES $<TARGET_FILE:zlibstatic>)
set(ZLIB_FOUND TRUE)

# libpng
set(SKIP_INSTALL_ALL ON)
set(PNG_BUILD_ZLIB ON CACHE INTERNAL "" FORCE)
set(PNG_STATIC ON CACHE INTERNAL "" FORCE)
set(PNG_TESTS OFF CACHE INTERNAL "" FORCE)
FetchContent_Declare(
  libpng
  GIT_REPOSITORY https://github.com/glennrp/libpng.git
  GIT_TAG v1.6.35
)
FetchContent_MakeAvailable(libpng)
add_dependencies(png_static zlibstatic)
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
