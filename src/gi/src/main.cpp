#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <vector>

#include "Mmap.h"
#include "Gi.h"
#include <Imgio.h>
#include <LogFlushListener.h>

namespace mx = MaterialX;
using namespace gtl;
using namespace doctest;

const static uint32_t REF_IMAGE_WIDTH = 512;
const static uint32_t REF_IMAGE_HEIGHT = 512;
const char* REF_IMAGE_EXT = ".png";

REGISTER_LISTENER("LogFlush", 1, GtLogFlushListener);

class GraphicalTestFixture
{
public:
  GraphicalTestFixture()
  {
    mx::DocumentPtr mtlxStdLib = mx::createDocument();
    mx::FilePathVec libFolders; // All directories if left empty.
    mx::loadLibraries(libFolders, { GI_MTLX_STDLIB_DIR }, mtlxStdLib);

    std::string mdlSearchPath = std::string(GI_MTLX_STDLIB_DIR) + "/mdl";

    GiInitParams params = {
      .shaderPath = GI_SHADER_SOURCE_DIR,
      .mdlRuntimePath = GI_MDL_LIB_DIR,
      .mdlSearchPaths = { mdlSearchPath },
      .mtlxStdLib = mtlxStdLib
    };

    REQUIRE_EQ(giInitialize(params), GiStatus::Ok);

    m_renderBuffer = giCreateRenderBuffer(REF_IMAGE_WIDTH, REF_IMAGE_HEIGHT);
    REQUIRE(m_renderBuffer);

    m_scene = giCreateScene();
    REQUIRE(m_scene);

    loadRefImage();
  }

  ~GraphicalTestFixture()
  {
    giDestroyScene(m_scene);
    giDestroyRenderBuffer(m_renderBuffer);
    giTerminate();
  }

protected:
  GiRenderBuffer* m_renderBuffer = nullptr;
  GiScene* m_scene = nullptr;

  ImgioImage m_refImage;

private:
  void loadRefImage()
  {
    std::string testName = doctest::detail::g_cs->currentTest->m_name;
    auto imgPath = std::string(GI_REF_IMAGE_DIR) + "/" + testName + REF_IMAGE_EXT;

    GiFile* file;
    REQUIRE(giFileOpen(imgPath.c_str(), GiFileUsage::Read, &file));

    size_t size = giFileSize(file);
    void* data = giMmap(file, 0, size);
    REQUIRE(data);

    ImgioImage img;
    REQUIRE_EQ(ImgioLoadImage(data, size, &img), ImgioError::None);

    giMunmap(file, data);
    giFileClose(file);

    REQUIRE_EQ(img.width, REF_IMAGE_WIDTH);
    REQUIRE_EQ(img.height, REF_IMAGE_HEIGHT);

    m_refImage = img;
  }

public:
  bool compareWithRef(const float* data)
  {
    size_t totalComponentCount = m_refImage.width * m_refImage.height * 4;

    for (size_t c = 0; c < totalComponentCount; c++)
    {
      uint8_t c1 = uint8_t(fminf(255.0f, data[c] * 255.0f));
      uint8_t c2 = m_refImage.data[c];

      if (c1 != c2)
      {
        return false;
      }
    }

    return true;
  }
};

TEST_CASE_FIXTURE(GraphicalTestFixture, "NoGeo")
{
  GiShaderCacheParams shaderCacheParams = {
    .aovId = GiAovId::Color,
    .depthOfField = false,
    .domeLightCameraVisible = false,
    .filterImportanceSampling = false,
    .materialCount = 0,
    .materials = nullptr,
    .nextEventEstimation = false,
    .progressiveAccumulation = true,
    .scene = m_scene
  };

  GiShaderCache* shaderCache = giCreateShaderCache(shaderCacheParams);
  REQUIRE(shaderCache);

  GiGeomCacheParams geomCacheParams = {
    .meshInstanceCount = 0,
    .meshInstances = nullptr,
    .shaderCache = shaderCache
  };

  GiGeomCache* geomCache = giCreateGeomCache(geomCacheParams);
  REQUIRE(geomCache);

  GiCameraDesc camDesc = {
    .position = { 0.0f, 0.0f, 0.0f },
    .forward = { 0.0f, 0.0f, -1.0f },
    .up = { 0.0f, 1.0f, 0.0f },
    .vfov = 1.57f,
    .fStop = 0.0f,
    .focusDistance = 0.0f,
    .focalLength = 0.0f,
    .clipStart = 0.0f,
    .clipEnd = FLT_MAX,
    .exposure = 1.0f
  };

  GiRenderParams renderParams =  {
    .camera = camDesc,
    .geomCache = geomCache,
    .shaderCache = shaderCache,
    .renderBuffer = m_renderBuffer,
    .lightIntensityMultiplier = 1.0f,
    .maxBounces = 8,
    .spp = 1,
    .rrBounceOffset = 255,
    .rrInvMinTermProb = 0.0f,
    .maxSampleValue = 100.0f,
    .backgroundColor = { 0.5f, 0.5f, 0.5f, 1.0f },
    .domeLight = nullptr,
    .scene = m_scene
  };

  std::vector<float> outputImg(REF_IMAGE_WIDTH * REF_IMAGE_HEIGHT * 4);
  REQUIRE_EQ(giRender(renderParams, outputImg.data()), GiStatus::Ok);

  CHECK(compareWithRef(outputImg.data()));

  giDestroyShaderCache(shaderCache);
  giDestroyGeomCache(geomCache);
}
