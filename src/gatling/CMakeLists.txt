add_executable(
  gatling
  main.cpp
  Argparse.h
  Argparse.cpp
  SimpleRenderTask.cpp
  SimpleRenderTask.h
)

target_link_libraries(
  gatling
  ar cameraUtil hd hf hgi hio usd usdGeom usdImaging
)
