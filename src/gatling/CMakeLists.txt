add_executable(
  gatling
  main.cpp
  Argparse.h
  Argparse.cpp
  SimpleRenderTask.cpp
  SimpleRenderTask.h
)

set_target_properties(
  gatling
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

target_link_libraries(
  gatling
  ar cameraUtil hd hf hgi hio usd usdGeom usdImaging
)
