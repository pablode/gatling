#pragma once

#include <pxr/pxr.h>

#include <string>

PXR_NAMESPACE_OPEN_SCOPE

class HdRenderDelegate;

struct AppSettings
{
  std::string sceneFilePath;
  std::string outputFilePath;
  int imageWidth;
  int imageHeight;
  std::string cameraPath;
  bool help;
};

bool ParseArgs(int argc, const char* argv[], HdRenderDelegate& renderDelegate, AppSettings& settings);

PXR_NAMESPACE_CLOSE_SCOPE
