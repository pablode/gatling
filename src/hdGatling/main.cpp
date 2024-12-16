//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#include "tokens.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <quill/Frontend.h>

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/gt/LogFlushListener.h>

#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/getenv.h>
#include <pxr/base/tf/setenv.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginHandle.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/task.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/utils.h>
#include <pxr/imaging/hf/pluginDesc.h>
#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hio/imageRegistry.h>
#include <pxr/imaging/hio/types.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/usd/usdRender/spec.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include <filesystem>

namespace fs = std::filesystem;
using namespace gtl;

PXR_NAMESPACE_USING_DIRECTIVE

namespace
{
  fs::path _GetTestInputDir()
  {
    const char* testName = doctest::detail::g_cs->currentTest->m_name;
    return fs::path(HDGATLING_TESTENV_DIR) / testName;
  }

  fs::path _GetTestOutputDir()
  {
    const char* testName = doctest::detail::g_cs->currentTest->m_name;
    return fs::path(HDGATLING_TEST_OUTPUT_DIR) / testName;
  }

  class _ErrorCheckSink final : public quill::Sink
  {
  public:
    void write_log(quill::MacroMetadata const*, uint64_t, std::string_view, std::string_view,
                   std::string const&, std::string_view, quill::LogLevel logLevel, std::string_view,
                   std::string_view, std::vector<std::pair<std::string, std::string>> const*,
                   std::string_view, std::string_view log_statement) override
    {
      if (logLevel == quill::LogLevel::Error)
      {
        m_errorCount++;
      }
    }

    void flush_sink() override {}

    void reset_error_count() { m_errorCount = 0; }

    uint32_t get_error_count() const { return m_errorCount; }

  private:
    uint32_t m_errorCount = 0;
  };

  class _SimpleRenderTask final : public HdTask
  {
  private:
    HdRenderPassSharedPtr m_renderPass;
    HdRenderPassStateSharedPtr m_renderPassState;
    const TfTokenVector m_renderTags = TfTokenVector(1, HdRenderTagTokens->geometry);

  public:
    _SimpleRenderTask(const HdRenderPassSharedPtr& renderPass, const HdRenderPassStateSharedPtr& renderPassState)
      : HdTask(SdfPath::EmptyPath())
      , m_renderPass(renderPass)
      , m_renderPassState(renderPassState)
    {
    }

    void Sync(HdSceneDelegate* sceneDelegate, HdTaskContext* taskContext, HdDirtyBits* dirtyBits) override
    {
      m_renderPass->Sync();
    }

    void Prepare(HdTaskContext* taskContext, HdRenderIndex* renderIndex) override
    {
      const HdResourceRegistrySharedPtr& resourceRegistry = renderIndex->GetResourceRegistry();
      m_renderPassState->Prepare(resourceRegistry);
    }

    void Execute(HdTaskContext* taskContext) override
    {
      m_renderPass->Execute(m_renderPassState, m_renderTags);
    }

    const TfTokenVector& GetRenderTags() const override { return m_renderTags; }
  };

  float _AccurateLinearToSrgb(float linearValue)
  {
    // Moving Frostbite to Physically Based Rendering 3.0, Section 5.1.5:
    // https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    float sRgbLo = linearValue * 12.92f;
    float sRgbHi = (std::pow(std::abs(linearValue), 1.0f / 2.4f) * 1.055f) - 0.055f;
    return (linearValue <= 0.0031308f) ? sRgbLo : sRgbHi;
  }

  HioImage::StorageSpec _MakeStorageSpec(uint32_t width, uint32_t height, void* data)
  {
    HioImage::StorageSpec spec;
    spec.width = (int) width;
    spec.height = (int) height;
    spec.depth = 1;
    spec.format = HioFormat::HioFormatUNorm8Vec4srgb;
    spec.flipped = true;
    spec.data = data;
    return spec;
  }

  struct _GraphicalTestPaths
  {
    fs::path testImg;
    fs::path refImg;
    fs::path diffImg;
  };

  _GraphicalTestPaths _MakeGraphicalTestPaths(const std::string& name)
  {
    std::string testImgName = "test";
    std::string refImgName = "ref";
    std::string diffImgName = "diff";

    if (!name.empty())
    {
      testImgName += "_" + name;
      refImgName += "_" + name;
      diffImgName += "_" + name;
    }

    return _GraphicalTestPaths {
      .testImg = _GetTestOutputDir() / (testImgName + ".png"),
      .refImg = _GetTestInputDir() / (refImgName + ".png"),
      .diffImg = _GetTestOutputDir() / (diffImgName + ".png"),
    };
  }
}

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  (HdGatlingRendererPlugin)
  (gtl)
);

TF_DEFINE_PRIVATE_TOKENS(
  _nsTokens,
  ((spp, "gtl:spp"))
  ((errorPixelThreshold, "gtl:errorPixelThreshold"))
  ((jitteredSampling, "gtl:jitteredSampling"))
  ((clippingPlanes, "gtl:clippingPlanes"))
);

int main(int argc, char** argv)
{
  // Find test installation before global installation
  TfSetenv("PXR_PLUGINPATH_NAME", HDGATLING_INSTALL_DIR);

  // Register plugin
  std::string plugInfoDir = GB_FMT("{}/hdGatling/resources", HDGATLING_INSTALL_DIR);

  PlugRegistry& plugRegistry = PlugRegistry::GetInstance();
  plugRegistry.RegisterPlugins(plugInfoDir);

  // Run tests
  doctest::Context context;
  context.applyCommandLine(argc, argv);

  int result = context.run();

  return result;
}

class GraphicalTestContext
{
private:
  std::unique_ptr<UsdImagingDelegate> m_sceneDelegate;
  HdRenderIndex* m_renderIndex;

  UsdStageRefPtr m_stage;
  UsdRenderSpec m_renderSpec;

  HdRenderDelegate* m_renderDelegate;
  HdRendererPluginHandle m_plugin;

  struct NamespacedSettings
  {
    uint32_t spp = 1;
    uint32_t errorPixelThreshold = 0;
    bool jitteredSampling = true;
    bool clippingPlanes = false;
  };

public:
  GraphicalTestContext(const fs::path& usdFilePath)
  {
    HdRendererPluginRegistry& pluginRegistry = HdRendererPluginRegistry::GetInstance();
    m_plugin = pluginRegistry.GetOrCreateRendererPlugin(_tokens->HdGatlingRendererPlugin);
    REQUIRE(m_plugin);
    REQUIRE(m_plugin->IsSupported());

    m_renderDelegate = m_plugin->CreateRenderDelegate();
    REQUIRE(m_renderDelegate);

    // Register listener after delegate initializes logger
    doctest::registerReporter<GtLogFlushListener>("LogFlush", 1, false);

    m_renderDelegate->SetRenderSetting(HdRenderSettingsTokens->enableInteractive, VtValue(false));

    m_stage = UsdStage::Open(usdFilePath.string());
    REQUIRE(m_stage);

    m_renderIndex = HdRenderIndex::New(m_renderDelegate, HdDriverVector());
    REQUIRE(m_renderIndex);

    m_sceneDelegate = std::make_unique<UsdImagingDelegate>(m_renderIndex, SdfPath::AbsoluteRootPath());
    m_sceneDelegate->Populate(m_stage->GetPseudoRoot());
    m_sceneDelegate->SetTime(0);
    m_sceneDelegate->SetRefineLevelFallback(4);

    SdfPath renderSettingsPrimPath;
    REQUIRE(m_stage->HasMetadata(UsdRenderTokens->renderSettingsPrimPath));
    m_stage->GetMetadata(UsdRenderTokens->renderSettingsPrimPath, &renderSettingsPrimPath);

    auto renderSettings = UsdRenderSettings(m_stage->GetPrimAtPath(renderSettingsPrimPath));
    REQUIRE(renderSettings);

    auto namespaces = TfTokenVector(1, _tokens->gtl);
    m_renderSpec = UsdRenderComputeSpec(renderSettings, namespaces);
  }

  ~GraphicalTestContext()
  {
    m_sceneDelegate.reset();
    m_stage.Reset();
    delete m_renderIndex;
    m_plugin->DeleteRenderDelegate(m_renderDelegate);
  }

private:
  void readNamespacedSettings(const VtDictionary& ns, NamespacedSettings& settings)
  {
    {
      auto it = ns.find(_nsTokens->spp);
      if (it != ns.end())
      {
        REQUIRE(it->second.IsHolding<int>());
        settings.spp = it->second.UncheckedGet<int>();
      }
    }
    {
      auto it = ns.find(_nsTokens->errorPixelThreshold);
      if (it != ns.end())
      {
        REQUIRE(it->second.IsHolding<int>());
        settings.errorPixelThreshold = it->second.UncheckedGet<int>();
      }
    }
    {
      auto it = ns.find(_nsTokens->jitteredSampling);
      if (it != ns.end())
      {
        REQUIRE(it->second.IsHolding<bool>());
        settings.jitteredSampling = it->second.UncheckedGet<bool>();
      }
    }
    {
      auto it = ns.find(_nsTokens->clippingPlanes);
      if (it != ns.end())
      {
        REQUIRE(it->second.IsHolding<bool>());
        settings.clippingPlanes = it->second.UncheckedGet<bool>();
      }
    }
  }

  void diffAgainstRef(const std::vector<uint8_t>& testValues,
                      uint32_t width, uint32_t height,
                      const fs::path& refPath,
                      const fs::path& diffPath,
                      uint32_t errorPixelThreshold)
  {
    fs::remove(diffPath);

    HioImageSharedPtr refImage = HioImage::OpenForReading(refPath.string());
    REQUIRE(refImage);

    REQUIRE_EQ(refImage->GetWidth(), width);
    REQUIRE_EQ(refImage->GetHeight(), height);
    REQUIRE_EQ(refImage->GetFormat(), HioFormat::HioFormatUNorm8Vec4srgb);

    int byteCount = width * height * 4;
    std::vector<uint8_t> refValues(byteCount);

    HioImage::StorageSpec refStorage = _MakeStorageSpec(width, height, refValues.data());
    REQUIRE(refImage->Read(refStorage));

    int errorPixelCount = 0;
    std::vector<uint8_t> diffValues(byteCount);
    for (uint32_t i = 0; i < byteCount; i++)
    {
      int diff = std::abs(int(refValues[i]) - int(testValues[i]));
      errorPixelCount += (diff > 0) ? 1 : 0;
      diffValues[i] = 255 - uint8_t(diff);
    }

    WARN_EQ(errorPixelCount, 0);
    if (errorPixelCount == 0)
    {
      return;
    }

    CHECK_LE(errorPixelCount, errorPixelThreshold);

    HioImageSharedPtr diffImage = HioImage::OpenForWriting(diffPath.string());
    REQUIRE(diffImage);

    VtDictionary metadata;
    HioImage::StorageSpec diffStorage = _MakeStorageSpec(width, height, diffValues.data());
    REQUIRE(diffImage->Write(diffStorage, metadata));
  }

  void produceProduct(const UsdRenderSpec::Product& product)
  {
    const auto& renderVarIndices = product.renderVarIndices;
    REQUIRE_EQ(renderVarIndices.size(), 1);
    const auto& renderVars = m_renderSpec.renderVars;
    REQUIRE(!renderVars.empty());

    const UsdRenderSpec::RenderVar& renderVar = renderVars[renderVarIndices[0]];

    // Set render settings.
    NamespacedSettings namespacedSettings;
    readNamespacedSettings(product.namespacedSettings, namespacedSettings);

    setRenderSetting(HdGatlingSettingsTokens->spp, VtValue(namespacedSettings.spp));
    setRenderSetting(HdGatlingSettingsTokens->depthOfField, VtValue(!product.disableDepthOfField));
    setRenderSetting(HdGatlingSettingsTokens->jitteredSampling, VtValue(namespacedSettings.jitteredSampling));
    setRenderSetting(HdGatlingSettingsTokens->clippingPlanes, VtValue(namespacedSettings.clippingPlanes));

    // Set up rendering state.
    uint32_t width = product.resolution[0];
    uint32_t height = product.resolution[1];

    auto camera = static_cast<HdCamera*>(m_renderIndex->GetSprim(HdTokens->camera, product.cameraPath));
    REQUIRE(camera);

    auto aovName = TfToken(renderVar.sourceName);
    auto aovDesc = m_renderDelegate->GetDefaultAovDescriptor(aovName);

    auto renderBuffer = static_cast<HdRenderBuffer*>(m_renderDelegate->CreateFallbackBprim(HdPrimTypeTokens->renderBuffer));
    REQUIRE(renderBuffer);
    REQUIRE(renderBuffer->Allocate(GfVec3i(width, height, 1), aovDesc.format, aovDesc.multiSampled));

    CameraUtilFraming framing;
    framing.dataWindow = GfRect2i(GfVec2i(0), product.resolution); // FIXME: consider product.dataWindowNDC
    framing.displayWindow = GfRange2f(GfVec2f(0.0f), GfVec2f(product.resolution + GfVec2i(1)));
    framing.pixelAspectRatio = product.pixelAspectRatio;

    auto renderPassState = std::make_shared<HdRenderPassState>();

    auto conformWindowPolicy = HdUtils::ToConformWindowPolicy(product.aspectRatioConformPolicy);
#if PXR_VERSION <= 2311
    bool useHdCameraWindowPolicy = false;
    std::pair<bool, CameraUtilConformWindowPolicy> overrideWindowPolicy(useHdCameraWindowPolicy, conformWindowPolicy);
    renderPassState->SetCameraAndFraming(camera, framing, overrideWindowPolicy);
#else
    std::optional<CameraUtilConformWindowPolicy> overrideWindowPolicy(conformWindowPolicy);
    renderPassState->SetCamera(camera);
    renderPassState->SetFraming(framing);
    renderPassState->SetOverrideWindowPolicy(overrideWindowPolicy);
#endif

    HdRenderPassAovBindingVector aovBindings(1);
    aovBindings[0].aovName = aovName;
    aovBindings[0].clearValue = aovDesc.clearValue;
    aovBindings[0].renderBuffer = renderBuffer;
    renderPassState->SetAovBindings(aovBindings);

    HdRprimCollection renderCollection(HdTokens->geometry, HdReprSelector(HdReprTokens->refined));
    HdRenderPassSharedPtr renderPass = m_renderDelegate->CreateRenderPass(m_renderIndex, renderCollection);
    REQUIRE(renderPass);

    HdTaskSharedPtrVector tasks;
    tasks.push_back(std::make_shared<_SimpleRenderTask>(renderPass, renderPassState));

    // Render, compare single frame.
    HdEngine engine;
    engine.Execute(m_renderIndex, &tasks);
    renderBuffer->Resolve();

    float* mappedMem = (float*) renderBuffer->Map();
    REQUIRE(mappedMem);

    bool gammaEncode = (aovBindings[0].aovName == HdAovTokens->color);
    bool isInt = (renderBuffer->GetFormat() == HdFormatInt32);
    int compCount = (renderBuffer->GetFormat() == HdFormatFloat32Vec4) ? 4 :
                      ((renderBuffer->GetFormat() == HdFormatFloat32Vec3) ? 3 : 1);

    size_t pixelCount = width * height;
    std::vector<uint8_t> byteValues(pixelCount * 4);
    for (int i = 0; i < pixelCount; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        if (isInt && compCount == 1)
        {
          int r = *((int*) &mappedMem[i * compCount]);
          byteValues[i * 4 + 0] = (r >>  0) & 0xFF;
          byteValues[i * 4 + 1] = (r >>  8) & 0xFF;
          byteValues[i * 4 + 2] = (r >> 16) & 0xFF;
          break;
        }

        float r = mappedMem[i * compCount + std::min(j, compCount - 1)];
        if (gammaEncode)
        {
          r =  _AccurateLinearToSrgb(r);
        }
        byteValues[i * 4 + j] = uint8_t(r * 255.0);
      }

      byteValues[i * 4 + 3] = 255; // always opaque
    }

    renderBuffer->Unmap();

    std::string productName = product.name.GetString();
    auto paths = _MakeGraphicalTestPaths(productName);

    fs::create_directories(paths.testImg.parent_path());
    HioImageSharedPtr image = HioImage::OpenForWriting(paths.testImg.string());
    REQUIRE(image);

    VtDictionary metadata;
    HioImage::StorageSpec testStorage = _MakeStorageSpec(width, height, byteValues.data());
    REQUIRE(image->Write(testStorage, metadata));

    diffAgainstRef(byteValues, width, height, paths.refImg, paths.diffImg, namespacedSettings.errorPixelThreshold);

    // Dispose of resources.
    HdRenderParam* renderParam = m_renderDelegate->GetRenderParam();
    REQUIRE(renderParam);
    renderBuffer->Finalize(renderParam);
    m_renderDelegate->DestroyBprim(renderBuffer);
  }

public:
  void performTest()
  {
    for (const UsdRenderSpec::Product product : m_renderSpec.products)
    {
      produceProduct(product);
    }
  }

  const UsdStageRefPtr& getStage()
  {
    return m_stage;
  }

  void setRenderSetting(TfToken name, VtValue value)
  {
    m_renderDelegate->SetRenderSetting(name, value);
  }
};

class GraphicalTestFixture
{
private:
  std::shared_ptr<_ErrorCheckSink> m_errorCheckSink;

public:
  GraphicalTestFixture()
  {
    auto sink = quill::Frontend::create_or_get_sink<_ErrorCheckSink>("ErrorCheck");
    m_errorCheckSink = std::static_pointer_cast<_ErrorCheckSink>(sink);
    m_errorCheckSink->reset_error_count();
    gbLogInit({ m_errorCheckSink });
  }

  ~GraphicalTestFixture()
  {
    CHECK_EQ(m_errorCheckSink->get_error_count(), 0);
  }
};

class SimpleGraphicalTestFixture : public GraphicalTestFixture
{
public:
  SimpleGraphicalTestFixture()
  {
    GraphicalTestContext context(_GetTestInputDir() / "scene.usd");
    context.performTest();
  }
};

TEST_SUITE("Graphical")
{
  TEST_CASE_FIXTURE(SimpleGraphicalTestFixture, "Materials.MtlxViewDirection")
  {
  }

  TEST_CASE_FIXTURE(SimpleGraphicalTestFixture, "Mesh.PrimvarInterpolation")
  {
  }

  TEST_CASE_FIXTURE(SimpleGraphicalTestFixture, "Render.AOVs")
  {
  }

  TEST_CASE_FIXTURE(SimpleGraphicalTestFixture, "Render.Empty1x1")
  {
  }
}
