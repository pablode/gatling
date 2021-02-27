#include <pxr/pxr.h>
#include "pxr/usd/ar/resolver.h"
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

HdRendererPluginHandle GetHdGatlingPlugin()
{
  HdRendererPluginRegistry& registry = HdRendererPluginRegistry::GetInstance();

  HfPluginDescVector pluginDescriptors;
  registry.GetPluginDescs(&pluginDescriptors);

  for (const HfPluginDesc& pluginDesc : pluginDescriptors)
  {
    const TfToken& pluginId = pluginDesc.id;

    if (pluginId != _AppTokens->HdGatlingRendererPlugin)
    {
      continue;
    }

    HdRendererPluginHandle plugin = registry.GetOrCreateRendererPlugin(pluginId);

    return plugin;
  }

  return HdRendererPluginHandle();
}

HdCamera* FindCamera(UsdStageRefPtr& stage, HdRenderIndex* renderIndex, std::string& settingsCameraPath)
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

  HdCamera* camera = (HdCamera*) dynamic_cast<HdCamera*>(renderIndex->GetSprim(HdTokens->camera, cameraPath));

  return camera;
}

int main(int argc, const char* argv[])
{
  // Init plugin.
  HdRendererPluginHandle pluginHandle = GetHdGatlingPlugin();

  if (!pluginHandle)
  {
    fprintf(stderr, "HdGatling plugin not found!\n");
    return EXIT_FAILURE;
  }

  if (!pluginHandle->IsSupported())
  {
    fprintf(stderr, "HdGatling plugin is not supported!\n");
    return EXIT_FAILURE;
  }

  HdRenderDelegate* renderDelegate = pluginHandle->CreateRenderDelegate();
  TF_VERIFY(renderDelegate);

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
  ArGetResolver().ConfigureResolverForAsset(settings.sceneFilePath);
  UsdStageRefPtr stage = UsdStage::Open(settings.sceneFilePath);

  if (!stage)
  {
    fprintf(stderr, "Unable to open USD stage file.\n");
    return EXIT_FAILURE;
  }

  HdRenderIndex* renderIndex = HdRenderIndex::New(renderDelegate, HdDriverVector());
  TF_VERIFY(renderIndex);

  UsdImagingDelegate frontend(renderIndex, SdfPath::AbsoluteRootPath());
  frontend.Populate(stage->GetPseudoRoot());
  frontend.SetTime(0);
  frontend.SetRefineLevelFallback(4);

  HdCamera* camera = FindCamera(stage, renderIndex, settings.cameraPath);
  if (!camera)
  {
    fprintf(stderr, "Camera not found!\n");
    return EXIT_FAILURE;
  }

  // Set up rendering context.
  HdRenderBuffer* renderBuffer = (HdRenderBuffer*) renderDelegate->CreateFallbackBprim(HdPrimTypeTokens->renderBuffer);
  renderBuffer->Allocate(GfVec3i(settings.imageWidth, settings.imageHeight, 1), HdFormatFloat32Vec4, false);

  HdRenderPassAovBindingVector aovBindings(1);
  aovBindings[0].aovName = HdAovTokens->color;
  aovBindings[0].renderBuffer = renderBuffer;

  CameraUtilFraming framing;
  framing.dataWindow = GfRect2i(GfVec2i(0, 0), GfVec2i(settings.imageWidth, settings.imageHeight));
  framing.displayWindow = GfRange2f(GfVec2f(0.0f, 0.0f), GfVec2f((float) settings.imageWidth, (float) settings.imageHeight));
  framing.pixelAspectRatio = 1.0f;

  std::pair<bool, CameraUtilConformWindowPolicy> overrideWindowPolicy(false, CameraUtilFit);

  auto renderPassState = std::make_shared<HdRenderPassState>();
  renderPassState->SetCameraAndFraming(camera, framing, overrideWindowPolicy);
  renderPassState->SetAovBindings(aovBindings);

  HdRprimCollection renderCollection(HdTokens->geometry, HdReprSelector(HdReprTokens->refined));
  HdRenderPassSharedPtr renderPass = renderDelegate->CreateRenderPass(renderIndex, renderCollection);

  TfTokenVector renderTags(1, HdRenderTagTokens->geometry);
  auto renderTask = std::make_shared<SimpleRenderTask>(renderPass, renderPassState, renderTags);

  HdTaskSharedPtrVector tasks;
  tasks.push_back(renderTask);

  // Perform rendering.
  HdEngine engine;

  engine.Execute(renderIndex, &tasks);

  // Write image to file.
  renderBuffer->Resolve();
  TF_VERIFY(renderBuffer->IsConverged());

  HioImageSharedPtr image = HioImage::OpenForWriting(settings.outputFilePath);

  if (!image)
  {
    fprintf(stderr, "Unable to open output file for writing!\n");
    return EXIT_FAILURE;
  }

  void* mappedMem = renderBuffer->Map();
  TF_VERIFY(mappedMem != nullptr);

  HioImage::StorageSpec storage;
  storage.width = (int) renderBuffer->GetWidth();
  storage.height = (int) renderBuffer->GetHeight();
  storage.depth = (int) renderBuffer->GetDepth();
  storage.format = HioFormat::HioFormatFloat32Vec4;
  storage.flipped = true;
  storage.data = mappedMem;

  VtDictionary metadata;
  image->Write(storage, metadata);

  renderBuffer->Unmap();

  renderDelegate->DestroyBprim(renderBuffer);

  return EXIT_SUCCESS;
}
