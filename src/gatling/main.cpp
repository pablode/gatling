//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include <pxr/pxr.h>
#include <pxr/base/gf/gamma.h>
#include <pxr/base/tf/stopwatch.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hf/pluginDesc.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/tokens.h>
#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hio/imageRegistry.h>
#include <pxr/imaging/hio/types.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include <algorithm>

#include "Argparse.h"
#include "SimpleRenderTask.h"

PXR_NAMESPACE_USING_DIRECTIVE

TF_DEFINE_PRIVATE_TOKENS(
  _AppTokens,
  (HdGatlingRendererPlugin)
);

namespace
{
  HdCamera* _FindCamera(UsdStageRefPtr& stage, HdRenderIndex* renderIndex, std::string& settingsCameraPath)
  {
    SdfPath cameraPath;

    if (!settingsCameraPath.empty())
    {
      cameraPath = SdfPath(settingsCameraPath);
    }
    else
    {
      UsdPrimRange primRange = stage->TraverseAll();
      for (auto prim = primRange.cbegin(); prim != primRange.cend(); prim++)
      {
        if (!prim->IsA<UsdGeomCamera>())
        {
          continue;
        }
        cameraPath = prim->GetPath();
        break;
      }
    }

    HdCamera* camera = static_cast<HdCamera*>(renderIndex->GetSprim(HdTokens->camera, cameraPath));

    return camera;
  }

  float _AccurateLinearToSrgb(float linearValue)
  {
    // Moving Frostbite to Physically Based Rendering 3.0, Section 5.1.5:
    // https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    float sRgbLo = linearValue * 12.92f;
    float sRgbHi = (std::pow(std::abs(linearValue), 1.0f / 2.4f) * 1.055f) - 0.055f;
    return (linearValue <= 0.0031308f) ? sRgbLo : sRgbHi;
  }
}

int main(int argc, const char* argv[])
{
  // Init plugin.
  HdRendererPluginRegistry& pluginRegistry = HdRendererPluginRegistry::GetInstance();
  HdRendererPluginHandle plugin = pluginRegistry.GetOrCreateRendererPlugin(_AppTokens->HdGatlingRendererPlugin);

  if (!plugin)
  {
    fprintf(stderr, "HdGatling plugin not found\n");
    return EXIT_FAILURE;
  }

  if (!plugin->IsSupported())
  {
    fprintf(stderr, "HdGatling plugin not supported\n");
    return EXIT_FAILURE;
  }

  HdRenderDelegate* renderDelegate = plugin->CreateRenderDelegate();
  TF_AXIOM(renderDelegate);

  // Handle cmdline args.
  AppSettings settings;
  if (!ParseArgs(argc, argv, *renderDelegate, settings))
  {
    return EXIT_FAILURE;
  }
  if (settings.help)
  {
    return EXIT_SUCCESS;
  }

  // Load scene.
  TfStopwatch loadTimer;
  loadTimer.Start();

  UsdStageRefPtr stage = UsdStage::Open(settings.sceneFilePath);

  loadTimer.Stop();

  if (!stage)
  {
    fprintf(stderr, "Unable to open USD stage file\n");
    return EXIT_FAILURE;
  }

  printf("USD scene loaded (%.3fs)\n", loadTimer.GetSeconds());
  fflush(stdout);

  HdRenderIndex* renderIndex = HdRenderIndex::New(renderDelegate, HdDriverVector());
  TF_AXIOM(renderIndex);

  std::unique_ptr<UsdImagingDelegate> sceneDelegate = std::make_unique<UsdImagingDelegate>(renderIndex, SdfPath::AbsoluteRootPath());
  sceneDelegate->Populate(stage->GetPseudoRoot());
  sceneDelegate->SetTime(0);
  sceneDelegate->SetRefineLevelFallback(4);

  HdCamera* camera = _FindCamera(stage, renderIndex, settings.cameraPath);
  if (!camera)
  {
    fprintf(stderr, "Camera not found\n");
    return EXIT_FAILURE;
  }

  // Set up rendering context.
  HdRenderBuffer* renderBuffer = (HdRenderBuffer*) renderDelegate->CreateFallbackBprim(HdPrimTypeTokens->renderBuffer);
  renderBuffer->Allocate(GfVec3i(settings.imageWidth, settings.imageHeight, 1), HdFormatFloat32Vec4, false);

  HdRenderPassAovBindingVector aovBindings(1);
  aovBindings[0].aovName = TfToken(settings.aov);
  aovBindings[0].renderBuffer = renderBuffer;

  CameraUtilFraming framing;
  framing.dataWindow = GfRect2i(GfVec2i(0, 0), GfVec2i(settings.imageWidth, settings.imageHeight));
  framing.displayWindow = GfRange2f(GfVec2f(0.0f, 0.0f), GfVec2f((float) settings.imageWidth, (float) settings.imageHeight));
  framing.pixelAspectRatio = 1.0f;

  auto renderPassState = std::make_shared<HdRenderPassState>();

#if PXR_VERSION <= 2311
  std::pair<bool, CameraUtilConformWindowPolicy> overrideWindowPolicy(false, CameraUtilFit);
  renderPassState->SetCameraAndFraming(camera, framing, overrideWindowPolicy);
#else
  std::optional<CameraUtilConformWindowPolicy> overrideWindowPolicy(CameraUtilFit);
  renderPassState->SetCamera(camera);
  renderPassState->SetFraming(framing);
  renderPassState->SetOverrideWindowPolicy(overrideWindowPolicy);
#endif

  renderPassState->SetAovBindings(aovBindings);

  HdRprimCollection renderCollection(HdTokens->geometry, HdReprSelector(HdReprTokens->refined));
  HdRenderPassSharedPtr renderPass = renderDelegate->CreateRenderPass(renderIndex, renderCollection);

  TfTokenVector renderTags(1, HdRenderTagTokens->geometry);
  auto renderTask = std::make_shared<SimpleRenderTask>(renderPass, renderPassState, renderTags);

  HdTaskSharedPtrVector tasks;
  tasks.push_back(renderTask);

  // Perform rendering.
  TfStopwatch renderTimer;
  renderTimer.Start();

  HdEngine engine;
  engine.Execute(renderIndex, &tasks);
  renderBuffer->Resolve();

  renderTimer.Stop();

  printf("Rendering finished (%.3fs)\n", renderTimer.GetSeconds());
  fflush(stdout);

  // Gamma correction.
  float* mappedMem = (float*) renderBuffer->Map();
  TF_AXIOM(mappedMem);

  if (settings.gammaCorrection)
  {
    int pixelCount = renderBuffer->GetWidth() * renderBuffer->GetHeight();
    for (int i = 0; i < pixelCount; i++)
    {
      mappedMem[i * 4 + 0] = _AccurateLinearToSrgb(mappedMem[i * 4 + 0]);
      mappedMem[i * 4 + 1] = _AccurateLinearToSrgb(mappedMem[i * 4 + 1]);
      mappedMem[i * 4 + 2] = _AccurateLinearToSrgb(mappedMem[i * 4 + 2]);
    }
  }

  // Write image to file.
  TfStopwatch writeTimer;
  writeTimer.Start();

  HioImageSharedPtr image = HioImage::OpenForWriting(settings.outputFilePath);

  if (!image)
  {
    fprintf(stderr, "Unable to open output file for writing\n");
    return EXIT_FAILURE;
  }

  HioImage::StorageSpec storage;
  storage.width = (int) renderBuffer->GetWidth();
  storage.height = (int) renderBuffer->GetHeight();
  storage.depth = (int) renderBuffer->GetDepth();
  storage.format = HioFormat::HioFormatFloat32Vec4;
  storage.flipped = true;
  storage.data = mappedMem;

  VtDictionary metadata;
  image->Write(storage, metadata);

  writeTimer.Stop();
  printf("Wrote image (%.3fs)\n", writeTimer.GetSeconds());
  fflush(stdout);

  renderBuffer->Unmap();
  HdRenderParam* renderParam = renderDelegate->GetRenderParam();
  renderBuffer->Finalize(renderParam);
  renderDelegate->DestroyBprim(renderBuffer);

  tasks.clear();
  renderTask.reset();
  renderPass.reset();
  sceneDelegate.reset();
  stage.Reset();
  delete renderIndex;
  plugin->DeleteRenderDelegate(renderDelegate);

  return EXIT_SUCCESS;
}
