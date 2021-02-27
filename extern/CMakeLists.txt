include(FetchContent)

# volk
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG 1.2.140
)
FetchContent_MakeAvailable(volk)
