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

#if !defined(NDEBUG) && !defined(GI_TEST_EXECUTABLE)
#define GI_SHADER_HOTLOADING
#endif

#include "Gi.h"
#include "TextureManager.h"
#include "Turbo.h"
#include "AssetReader.h"
#include "GlslShaderGen.h"
#include "MeshProcessing.h"
#include "interface/rp_main.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <atomic>
#include <optional>
#include <mutex>
#include <assert.h>

#include <gtl/ggpu/Stager.h>
#include <gtl/ggpu/BumpAllocator.h>
#include <gtl/ggpu/DeleteQueue.h>
#include <gtl/ggpu/DenseDataStore.h>
#include <gtl/ggpu/ResizableBuffer.h>
#include <gtl/cgpu/Cgpu.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#ifdef GI_SHADER_HOTLOADING
#include <efsw/efsw.hpp>
#endif
#include <gtl/mc/Material.h>
#include <gtl/mc/Frontend.h>
#include <gtl/mc/Runtime.h>
#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/Enum.h>
#include <gtl/gb/SmallVector.h>

#include <MaterialXCore/Document.h>

#include <blosc2.h>

#include <offsetAllocator.hpp>

#define GI_FATAL(msg)                               \
  do {                                              \
    GB_ERROR("{}:{}: {}", __FILE__, __LINE__, msg); \
    exit(EXIT_FAILURE);                             \
  } while (false)

namespace mx = MaterialX;

namespace gtl
{
  constexpr static const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

  namespace rp = shader_interface::rp_main;

  class McRuntime;

  struct GiGpuBufferView
  {
    uint64_t offset;
    uint64_t size;
  };

  struct GiMeshGpuData
  {
    CgpuBlas blas;
    CgpuBuffer payloadBuffer;
    rp::BlasPayload payload;
  };

  struct GiBvh
  {
    CgpuBuffer blasPayloadsBuffer;
    CgpuBuffer instanceIdsBuffer;
    GiScene*   scene;
    CgpuTlas   tlas;
  };

  struct GiImageBinding
  {
    GiImagePtr image;
    uint32_t index;
  };

  struct GiShaderCache
  {
    uint32_t                       aovMask;
    std::array<CgpuBindSet, 3>     bindSets;
    bool                           domeLightCameraVisible;
    std::vector<GiImageBinding>    imageBindings;
    std::vector<const GiMaterial*> materials;
    uint32_t                       maxTextureIndex;
    std::vector<CgpuShader>        missShaders;
    CgpuPipeline                   pipeline;
    CgpuShader                     rgenShader;
    GiScene*                       scene;
    bool                           resetSampleOffset = true;
  };

  struct GiMaterialGpuData
  {
    CgpuShader closestHit;
    std::vector<CgpuShader> anyHits; // optional
    OffsetAllocator::Allocation texOffsetAllocation;
    std::vector<GiImagePtr> images;
  };

  struct GiMaterial
  {
    McMaterial* mcMat = nullptr;
    std::string name;
    GiScene* scene;
    std::optional<GiMaterialGpuData> gpuData;
  };

  struct GiMesh
  {
    glm::mat3x4 transform;
    bool doubleSided;
    bool flipFacing;
    int id;
    std::vector<glm::mat3x4> instanceTransforms;
    std::vector<int> instanceIds;
    std::vector<GiPrimvarData> instancerPrimvars; // TODO: compress with blosc
    GiMaterial* material = nullptr;
    GiScene* scene;
    GiMeshData cpuData;
    std::optional<GiMeshGpuData> gpuData;
    bool visible = true;
    std::string name;
    uint32_t maxFaceId;
  };

  struct GiSphereLight
  {
    GiScene* scene;
    uint64_t gpuHandle;
  };

  struct GiDistantLight
  {
    GiScene* scene;
    uint64_t gpuHandle;
  };

  struct GiRectLight
  {
    GiScene* scene;
    uint64_t gpuHandle;
  };

  struct GiDiskLight
  {
    GiScene* scene;
    uint64_t gpuHandle;
  };

  struct GiDomeLight
  {
    GiScene* scene;
    std::string textureFilePath;
    glm::quat rotation;
    glm::vec3 baseEmission;
    float diffuse = 1.0f;
    float specular = 1.0f;
  };

  enum class GiSceneDirtyFlags : uint32_t
  {
    Clean                   = (1 << 0),
    DirtyBvh                = (1 << 1),
    DirtyFramebuffer        = (1 << 2),
    DirtyShadersRgen        = (1 << 3),
    DirtyShadersHit         = (1 << 4),
    DirtyShadersMiss        = (1 << 5),
    DirtyShadersAll         = (DirtyShadersRgen | DirtyShadersHit | DirtyShadersMiss),
    DirtyPipeline           = (1 << 6),
    DirtyAovBindingDefaults = (1 << 7),
    DirtyBindSets           = (1 << 8),
    All                     = ~0u
  };
  GB_DECLARE_ENUM_BITOPS(GiSceneDirtyFlags)

  struct GiScene
  {
    OffsetAllocator::Allocation domeLightsAllocation;
    GgpuDenseDataStore sphereLights;
    GgpuDenseDataStore distantLights;
    GgpuDenseDataStore rectLights;
    GgpuDenseDataStore diskLights;
    GiImagePtr domeLightTexture;
    GiDomeLight* domeLight = nullptr; // weak ptr
    glm::vec4 backgroundColor = glm::vec4(-1.0f); // used to initialize fallback dome light
    CgpuImage fallbackDomeLightTexture;
    std::unordered_set<GiMesh*> meshes;
    std::unordered_set<GiMaterial*> materials;
    std::mutex mutex;
    GiSceneDirtyFlags dirtyFlags = GiSceneDirtyFlags::All;
    GiShaderCache* shaderCache = nullptr;
    GiBvh* bvh = nullptr;
    GiRenderParams oldRenderParams = {};
    CgpuBuffer aovDefaultValues;
    uint32_t sampleOffset = 0;
    OffsetAllocator::Allocator texAllocator{rp::MAX_TEXTURE_COUNT};
  };

  struct GiRenderBuffer
  {
    CgpuBuffer deviceMem;
    CgpuBuffer hostMem;
    void* mappedHostMem = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t size = 0;
  };

  CgpuContext* s_ctx = nullptr;
  CgpuDeviceFeatures s_ctxFeatures;
  CgpuDeviceProperties s_ctxProperties;
  CgpuSampler s_texSampler;
  std::unique_ptr<GgpuStager> s_stager;
  std::mutex s_resourceDestroyerMutex;
  std::unique_ptr<GgpuDeleteQueue> s_deleteQueue;
  std::unique_ptr<GiGlslShaderGen> s_shaderGen;
  std::unique_ptr<McRuntime> s_mcRuntime;
  std::unique_ptr<McFrontend> s_mcFrontend;
  std::unique_ptr<GiMmapAssetReader> s_mmapAssetReader;
  std::unique_ptr<GiAggregateAssetReader> s_aggregateAssetReader;
  std::unique_ptr<GiTextureManager> s_texSys;
  std::shared_ptr<GgpuBumpAllocator> s_bumpAlloc;
  std::atomic_bool s_forceShaderCacheInvalid = false;
  std::atomic_bool s_resetSampleOffset = false;

#ifdef GI_SHADER_HOTLOADING
  class ShaderFileListener : public efsw::FileWatchListener
  {
  public:
    void handleFileAction([[maybe_unused]] efsw::WatchID watchId, [[maybe_unused]] const std::string& dir,
                          [[maybe_unused]] const std::string& filename, efsw::Action action,
                          [[maybe_unused]] std::string oldFilename) override
    {
      switch (action)
      {
      case efsw::Actions::Delete:
      case efsw::Actions::Modified:
      case efsw::Actions::Moved:
        s_forceShaderCacheInvalid = true;
        s_resetSampleOffset = true;
        break;
      default:
        break;
      }
    }
  };

  std::unique_ptr<efsw::FileWatcher> s_fileWatcher;
  ShaderFileListener s_shaderFileListener;
#endif

  glm::vec2 _EncodeOctahedral(glm::vec3 v)
  {
    v /= (fabsf(v.x) + fabsf(v.y) + fabsf(v.z));
    glm::vec2 ps = glm::vec2(v.x >= 0.0f ? +1.0f : -1.0f, v.y >= 0.0f ? +1.0f : -1.0f);
    return (v.z < 0.0f) ? ((1.0f - glm::abs(glm::vec2(v.y, v.x))) * ps) : glm::vec2(v.x, v.y);
  }

  uint32_t _EncodeDirection(glm::vec3 v)
  {
    v = glm::normalize(v);
    glm::vec2 e = _EncodeOctahedral(v);
    e = e * 0.5f + 0.5f;
    return glm::packUnorm2x16(e);
  }

  uint32_t _GiRenderBufferFormatStride(GiRenderBufferFormat format)
  {
    switch (format)
    {
    case GiRenderBufferFormat::Int32:
      return 4;
    case GiRenderBufferFormat::Float32:
      return 4;
    case GiRenderBufferFormat::Float32Vec4:
      return 4 * 4;
    default:
      assert(false);
      return 0;
    }
  }

  void _PrintInitInfo(const GiInitParams& params)
  {
    GB_LOG("gatling {}.{}.{} built against MaterialX {}.{}.{}", GI_VERSION_MAJOR, GI_VERSION_MINOR, GI_VERSION_PATCH,
                                                                MATERIALX_MAJOR_VERSION, MATERIALX_MINOR_VERSION, MATERIALX_BUILD_VERSION);
    GB_LOG("> shader path: \"{}\"", params.shaderPath);
    GB_LOG("> MDL runtime path: \"{}\"", params.mdlRuntimePath);
    GB_LOG("> MDL search paths: {}", params.mdlSearchPaths);
  }

  void _EncodeRenderBufferAsHeatmap(GiRenderBuffer* renderBuffer)
  {
    int channelCount = renderBuffer->width * renderBuffer->height * 4;
    float* rgbaImg = (float*) renderBuffer->mappedHostMem;

    float maxValue = 0.0f;
    for (int i = 0; i < channelCount; i += 4) {
      maxValue = std::max(maxValue, rgbaImg[i]); // only consider first channel
    }
    for (int i = 0; i < channelCount && maxValue > 0.0f; i += 4) {
      int valIndex = std::min(int((rgbaImg[i] / maxValue) * 255.0), 255);
      rgbaImg[i + 0] = (float) TURBO_SRGB_FLOATS[valIndex][0];
      rgbaImg[i + 1] = (float) TURBO_SRGB_FLOATS[valIndex][1];
      rgbaImg[i + 2] = (float) TURBO_SRGB_FLOATS[valIndex][2];
      rgbaImg[i + 3] = 255;
    }
  }

  // IMPORTANT: this needs to match the rp_main* shaders. It is asserted in cgpu.
  uint32_t _GetRpMainMaxRayHitAttributeSize()
  {
    return 8;
  }

  // IMPORTANT: this needs to match the rp_main* shaders. It is asserted in cgpu.
  uint32_t _GetRpMainMaxRayPayloadSize(uint32_t mediumStackSize)
  {
    uint32_t size = 84;
    if (mediumStackSize > 0)
    {
      size += mediumStackSize * 40 + 12;
    }
    return size;
  }

  GiStatus giInitialize(const GiInitParams& params)
  {
#ifdef NDEBUG
    std::string_view shaderPath = params.shaderPath;
#else
    // Use shaders dir in source tree for auto-reloading
    std::string_view shaderPath = GI_SHADER_SOURCE_DIR;
#endif

    mx::DocumentPtr mtlxStdLib = std::static_pointer_cast<mx::Document>(params.mtlxStdLib);
    if (!mtlxStdLib)
    {
      return GiStatus::Error;
    }

    gbLogInit();

    _PrintInitInfo(params);

    s_ctx = cgpuCreateContext("gatling", GI_VERSION_MAJOR, GI_VERSION_MINOR, GI_VERSION_PATCH);
    if (!s_ctx)
      goto fail;

    s_ctxFeatures = cgpuGetDeviceFeatures(s_ctx);
    s_ctxProperties = cgpuGetDeviceProperties(s_ctx);

    if (!cgpuCreateSampler(s_ctx, {
                            .addressModeU = CgpuSamplerAddressMode::Repeat,
                            .addressModeV = CgpuSamplerAddressMode::Repeat,
                            .addressModeW = CgpuSamplerAddressMode::Repeat
                          }, &s_texSampler))
    {
      goto fail;
    }

    s_stager = std::make_unique<GgpuStager>(s_ctx);
    if (!s_stager->allocate())
    {
      goto fail;
    }

    s_deleteQueue = std::make_unique<GgpuDeleteQueue>(s_ctx);

    s_mcRuntime = std::unique_ptr<McRuntime>(McLoadRuntime(params.mdlRuntimePath, params.mdlSearchPaths));
    if (!s_mcRuntime)
    {
      goto fail;
    }

    s_mcFrontend = std::make_unique<McFrontend>(mtlxStdLib, params.mtlxCustomNodesPath, *s_mcRuntime);

    s_shaderGen = std::make_unique<GiGlslShaderGen>();
    if (!s_shaderGen->init(shaderPath, *s_mcRuntime))
    {
      goto fail;
    }

    constexpr static uint32_t BUMP_ALLOC_SIZE = CGPU_MIN_UNIFORM_BUFFER_SIZE;
    s_bumpAlloc = GgpuBumpAllocator::make(s_ctx, *s_deleteQueue, BUMP_ALLOC_SIZE);

    if (!s_bumpAlloc)
    {
      goto fail;
    }

    s_mmapAssetReader = std::make_unique<GiMmapAssetReader>();
    s_aggregateAssetReader = std::make_unique<GiAggregateAssetReader>();
    s_aggregateAssetReader->addAssetReader(s_mmapAssetReader.get());

    s_texSys = std::make_unique<GiTextureManager>(s_ctx, *s_aggregateAssetReader, *s_stager, *s_deleteQueue);

#ifdef GI_SHADER_HOTLOADING
    s_fileWatcher = std::make_unique<efsw::FileWatcher>();
    s_fileWatcher->addWatch(shaderPath.data(), &s_shaderFileListener, true);
    s_fileWatcher->watch();
#endif

    blosc2_init();
    blosc2_set_nthreads(4);

    return GiStatus::Ok;

fail:
    giTerminate();

    return GiStatus::Error;
  }

  void giTerminate()
  {
    GB_LOG("terminating...");
  #ifdef GI_SHADER_HOTLOADING
    s_fileWatcher.reset();
  #endif
    s_bumpAlloc.reset();
    s_aggregateAssetReader.reset();
    s_mmapAssetReader.reset();
    if (s_texSys)
    {
      s_texSys->destroy();
      s_texSys.reset();
    }
    s_shaderGen.reset();
    if (s_stager)
    {
      s_stager->flush();
      s_stager->free();
      s_stager.reset();
    }
    if (s_texSampler.handle)
    {
      cgpuDestroySampler(s_ctx, s_texSampler);
      s_texSampler = {};
    }
    if (s_deleteQueue)
    {
      s_deleteQueue->destroyAll();
      s_deleteQueue.reset();
    }
    if (s_ctx)
    {
      cgpuDestroyContext(s_ctx);
      s_ctx = nullptr;
    }
    s_mcFrontend.reset();
    s_mcRuntime.reset();
  }

  void giRegisterAssetReader(GiAssetReader* reader)
  {
    s_aggregateAssetReader->addAssetReader(reader);
  }

  GiMaterial* giCreateMaterialFromMtlxStr(GiScene* scene, const char* name, const char* mtlxSrc)
  {
    McMaterial* mcMat = s_mcFrontend->createFromMtlxStr(mtlxSrc);
    if (!mcMat)
    {
      return nullptr;
    }

    GiMaterial* mat = new GiMaterial {
      .mcMat = mcMat,
      .name = name,
      .scene = scene
    };

    {
      std::lock_guard guard(scene->mutex);
      scene->materials.insert(mat);
    }
    return mat;
  }

  GiMaterial* giCreateMaterialFromMtlxDoc(GiScene* scene, const char* name, const std::shared_ptr<void/*MaterialX::Document*/> doc)
  {
    mx::DocumentPtr resolvedDoc = std::static_pointer_cast<mx::Document>(doc);
    if (!doc)
    {
      return nullptr;
    }

    McMaterial* mcMat = s_mcFrontend->createFromMtlxDoc(resolvedDoc);
    if (!mcMat)
    {
      return nullptr;
    }

    GiMaterial* mat = new GiMaterial {
      .mcMat = mcMat,
      .name = name,
      .scene = scene
    };

    {
      std::lock_guard guard(scene->mutex);
      scene->materials.insert(mat);
    }
    return mat;
  }

  GiMaterial* giCreateMaterialFromMdlFile(GiScene* scene, const char* name, const char* filePath, const char* subIdentifier, const GiMaterialParameters& params)
  {
    McMaterial* mcMat = s_mcFrontend->createFromMdlFile(filePath, subIdentifier, params);
    if (!mcMat)
    {
      return nullptr;
    }

    GiMaterial* mat = new GiMaterial {
      .mcMat = mcMat,
      .name = name,
      .scene = scene
    };

    {
      std::lock_guard guard(scene->mutex);
      scene->materials.insert(mat);
    }
    return mat;
  }

  void giDestroyMaterialGpuData(GiScene* scene, GiMaterialGpuData& gpuData)
  {
    if (gpuData.texOffsetAllocation.offset != OffsetAllocator::Allocation::NO_SPACE)
    {
      scene->texAllocator.free(gpuData.texOffsetAllocation);
    }
    if (gpuData.closestHit.handle)
    {
      cgpuDestroyShader(s_ctx, gpuData.closestHit);
    }
    for (CgpuShader shader : gpuData.anyHits)
    {
      cgpuDestroyShader(s_ctx, shader);
    }
  }

  void giDestroyMaterial(GiMaterial* mat)
  {
    GiScene* scene = mat->scene;
    {
      std::lock_guard guard(scene->mutex);

      if (mat->gpuData)
      {
        giDestroyMaterialGpuData(scene, *mat->gpuData);
      }

      scene->materials.erase(mat);

      for (GiMesh* m : scene->meshes)
      {
        if (m->material == mat)
        {
          m->material = nullptr;
        }
      }
    }

    delete mat->mcMat;
    delete mat;
  }

  uint64_t giAlignBuffer(uint64_t alignment, uint64_t bufferSize, uint64_t* totalSize)
  {
    if (bufferSize == 0)
    {
      return *totalSize;
    }

    const uint64_t offset = ((*totalSize) + alignment - 1) / alignment * alignment;

    (*totalSize) = offset + bufferSize;

    return offset;
  }

  GiMesh* giCreateMesh(GiScene* scene, const GiMeshDesc& desc)
  {
    GiMesh* mesh = new GiMesh {
      .transform = glm::mat3x4(1.0f),
      .doubleSided = desc.isDoubleSided,
      .flipFacing = desc.isLeftHanded,
      .id = desc.id,
      .scene = scene,
      .cpuData = giProcessMeshData(desc.faces, desc.faceIds, desc.vertices, desc.primvars),
      .name = desc.name,
      .maxFaceId = desc.maxFaceId
    };

    {
      std::lock_guard guard(scene->mutex);
      scene->meshes.insert(mesh);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBvh;
    }
    return mesh;
  }

  void giSetMeshTransform(GiMesh* mesh, const float* transform)
  {
    mesh->transform = glm::mat3x4(glm::transpose(glm::make_mat4(transform)));

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBvh;
    }
  }

  void giSetMeshInstanceTransforms(GiMesh* mesh, uint32_t count, const float (*transforms)[4][4])
  {
    mesh->instanceTransforms.resize(count);
    for (uint32_t i = 0; i < count; i++)
    {
      mesh->instanceTransforms[i] = glm::mat3x4(glm::transpose(glm::make_mat4((const float*) transforms[i])));
    }

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBvh;
    }
  }

  void giSetMeshInstanceIds(GiMesh* mesh, uint32_t count, int* ids)
  {
    mesh->instanceIds.resize(count);
    memcpy(mesh->instanceIds.data(), ids, count * sizeof(int));

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
    }
  }

  void giDestroyMeshGpuData(GiMeshGpuData& gpuData)
  {
    std::lock_guard guard(s_resourceDestroyerMutex); // Hydra sync is parallel
    s_deleteQueue->pushBack(gpuData.blas, gpuData.payloadBuffer);
  }

  void giSetMeshInstancerPrimvars(GiMesh* mesh, const std::vector<GiPrimvarData>& instancerPrimvars)
  {
    mesh->instancerPrimvars = instancerPrimvars;

    if (mesh->gpuData.has_value())
    {
      giDestroyMeshGpuData(*mesh->gpuData);
      mesh->gpuData.reset();
    }

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBvh;
    }
  }

  void giSetMeshMaterial(GiMesh* mesh, GiMaterial* mat)
  {
    McMaterial* newMcMat = mat->mcMat;
    McMaterial* oldMcMat = mesh->material ? mesh->material->mcMat : nullptr;

    bool transparencyChanged = bool(newMcMat) ^ bool(oldMcMat);
    bool primvarsChanged = bool(newMcMat) ^ bool(oldMcMat);

    GiSceneDirtyFlags dirtyFlags = GiSceneDirtyFlags::DirtyPipeline;
    if (oldMcMat && newMcMat)
    {
      // material data such as alpha is used in the BVH build
      transparencyChanged |= (newMcMat->hasCutoutTransparency != oldMcMat->hasCutoutTransparency);

      const auto& newPrimvarNames = newMcMat->sceneDataNames;
      const auto& oldPrimvarNames = oldMcMat->sceneDataNames;

      primvarsChanged |= newPrimvarNames.size() != oldPrimvarNames.size();

      if (!primvarsChanged)
      {
        for (size_t i = 0; i < newPrimvarNames.size(); i++)
        {
          if (strcmp(newPrimvarNames[i], oldPrimvarNames[i]) != 0)
          {
            primvarsChanged = true;
            break;
          }
        }
      }
    }

    if (transparencyChanged || primvarsChanged)
    {
      dirtyFlags |= GiSceneDirtyFlags::DirtyBvh;

      if (mesh->gpuData.has_value())
      {
        giDestroyMeshGpuData(*mesh->gpuData);
        mesh->gpuData.reset();
      }
    }

    mesh->material = mat;

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= dirtyFlags;
    }
  }

  void giSetMeshVisibility(GiMesh* mesh, bool visible)
  {
    mesh->visible = visible;

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBvh;
    }
  }

  void giDestroyMesh(GiMesh* mesh)
  {
    auto& gpuData = mesh->gpuData;

    if (gpuData.has_value())
    {
      giDestroyMeshGpuData(*gpuData);
      gpuData.reset();
    }

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->meshes.erase(mesh);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBvh;
    }
    delete mesh;
  }

  void _giBuildGeometryStructures(GiScene* scene,
                                  const GiShaderCache* shaderCache,
                                  std::vector<CgpuBlasInstance>& blasInstances,
                                  std::vector<rp::BlasPayload>& blasPayloads,
                                  std::vector<int>& instanceIds,
                                  uint64_t& totalIndicesSize,
                                  uint64_t& totalVerticesSize)
  {
    size_t meshCount = scene->meshes.size();
    blasInstances.reserve(meshCount);
    blasPayloads.reserve(meshCount);
    instanceIds.reserve(meshCount);

    for (auto it = scene->meshes.begin(); it != scene->meshes.end(); ++it)
    {
      GiMesh* mesh = *it;

      // Don't build BLAS/TLAS for non-visible geometry
      if (!mesh->visible)
      {
        continue;
      }

      // Find material for SBT index (FIXME: find a better solution)
      const GiMaterial* material = mesh->material;

      uint32_t materialIndex = UINT32_MAX;
      for (uint32_t i = 0; i < shaderCache->materials.size(); i++)
      {
          if (shaderCache->materials[i] == material)
          {
              materialIndex = i;
              break;
          }
      }
      if (materialIndex == UINT32_MAX)
      {
          GB_ERROR("invalid BLAS material");
          continue;
      }

      // Build mesh BLAS & buffers if they don't exist yet
      if (!mesh->gpuData.has_value())
      {
        std::vector<GiFace> meshFaces;
        std::vector<int> meshFaceIds;
        std::vector<GiVertex> meshVertices;
        std::vector<GiPrimvarData> meshPrimvars;
        giDecompressMeshData(mesh->cpuData, meshFaces, meshFaceIds, meshVertices, meshPrimvars);

        if (meshFaces.empty())
        {
          continue;
        }

        // Collect vertices
        std::vector<rp::FVertex> vertexData;
        std::vector<float> positionData;
        vertexData.resize(meshVertices.size());
        positionData.resize(meshVertices.size() * 3);

        for (uint32_t i = 0; i < meshVertices.size(); i++)
        {
          const GiVertex& cpuVert = meshVertices[i];
          uint32_t encodedNormal = _EncodeDirection(glm::make_vec3(cpuVert.norm));
          uint32_t encodedTangent = _EncodeDirection(glm::make_vec3(cpuVert.tangent));

          vertexData[i] = rp::FVertex{
            .field1 = { glm::make_vec3(cpuVert.pos), cpuVert.bitangentSign },
            .field2 = { *((float*) &encodedNormal), *((float*) &encodedTangent), cpuVert.u, cpuVert.v }
          };

          positionData[i * 3 + 0] = cpuVert.pos[0];
          positionData[i * 3 + 1] = cpuVert.pos[1];
          positionData[i * 3 + 2] = cpuVert.pos[2];
        }

        uint64_t verticesSize = vertexData.size() * sizeof(rp::FVertex);

        // Collect indices
        std::vector<uint32_t> indexData;
        indexData.reserve(meshFaces.size() * 3);

        for (uint32_t i = 0; i < meshFaces.size(); i++)
        {
          const auto* face = &meshFaces[i];
          indexData.push_back(face->v_i[0]);
          indexData.push_back(face->v_i[1]);
          indexData.push_back(face->v_i[2]);
        }

        uint64_t indicesSize = indexData.size() * sizeof(uint32_t);

        // Collect face IDs
        uint32_t faceIdStride = mesh->maxFaceId <= UINT8_MAX ? 1 : (mesh->maxFaceId <= UINT16_MAX ? 2 : 4);
        uint64_t faceIdsSize = (meshFaceIds.size() * faceIdStride + 3) / 4 * 4; // align buffer size to 4 bytes

        std::vector<uint8_t> faceIdData(faceIdsSize);
        for (size_t i = 0; i < meshFaceIds.size(); i++)
        {
          memcpy(&faceIdData[i * faceIdStride], &meshFaceIds[i], faceIdStride);
        }

        // Prepare scene data & preamble
        std::vector<const GiPrimvarData*> primvars;
        if (material)
        {
          const std::vector<const char*> sceneDataNames = material->mcMat->sceneDataNames;

          size_t sceneDataCount = sceneDataNames.size();
          int overflowCount = int(sceneDataCount) - int(rp::MAX_SCENE_DATA_COUNT);

          if (overflowCount > 0)
          {
              GB_ERROR("max scene data count exceeded for {}; ignoring {} scene data", mesh->name, overflowCount);
              sceneDataCount = rp::MAX_SCENE_DATA_COUNT;
          }

          primvars.resize(sceneDataCount, nullptr);

          if (sceneDataCount > 0)
          {
            GB_DEBUG("scene data for mesh {} with material {}:", mesh->name, material->name);
          }
          for (size_t i = 0; i < sceneDataCount; i++)
          {
            const char* sceneDataName = sceneDataNames[i];

            // FIXME: we should check if scene data and primvar types match to prevent crashes
            const GiPrimvarData* primvar = nullptr;
            for (const GiPrimvarData& p : mesh->instancerPrimvars)
            {
              if (p.name == sceneDataName && !p.data.empty())
              {
                primvar = &p;
                break;
              }
            }
            for (const GiPrimvarData& p : meshPrimvars) // override instancer primvars
            {
              if (p.name == sceneDataName && !p.data.empty())
              {
                primvar = &p;
                break;
              }
            }

            if (!primvar)
            {
              GB_DEBUG("> [{}] {} (not found!)", i, sceneDataName);
              continue;
            }

            GB_DEBUG("> [{}] {}", i, sceneDataName);
            primvars[i] = primvar;
          }
        }

        uint64_t preambleSize = sizeof(rp::BlasPayloadBufferPreamble);

        uint64_t payloadBufferSize = preambleSize;
        uint64_t indexBufferOffset = giAlignBuffer(sizeof(rp::FVertex), indicesSize, &payloadBufferSize);
        uint64_t vertexBufferOffset = giAlignBuffer(sizeof(rp::FVertex), verticesSize, &payloadBufferSize);
        uint64_t faceIdsBufferOffset = giAlignBuffer(sizeof(int), faceIdsSize, &payloadBufferSize);

        rp::BlasPayloadBufferPreamble preamble
        {
          .objectId = mesh->id,
          .faceIdsInfo = (faceIdStride << rp::FACE_ID_STRIDE_OFFSET) | uint32_t(faceIdsBufferOffset)
        };

        std::vector<uint32_t> sceneDataOffsets(primvars.size());
        for (size_t i = 0; i < primvars.size(); i++)
        {
          const GiPrimvarData* primvar = primvars[i];

          if (!primvar)
          {
            preamble.sceneDataInfos[i] = rp::SCENE_DATA_INVALID;
            continue;
          }

          uint64_t newPayloadBufferSize = payloadBufferSize;
          uint64_t sceneDataOffset = giAlignBuffer(rp::SCENE_DATA_ALIGNMENT, primvar->data.size(), &newPayloadBufferSize);

          if (sceneDataOffset >= UINT32_MAX)
          {
            GB_ERROR("scene data too large");
            preamble.sceneDataInfos[i] = rp::SCENE_DATA_INVALID;
            continue;
          }

          if ((sceneDataOffset & rp::SCENE_DATA_OFFSET_MASK) != sceneDataOffset || sceneDataOffset == rp::SCENE_DATA_OFFSET_MASK)
          {
            GB_ERROR("max scene data offset exceeded");
            preamble.sceneDataInfos[i] = rp::SCENE_DATA_INVALID;
            continue;
          }

          payloadBufferSize = newPayloadBufferSize;
          sceneDataOffsets[i] = uint32_t(sceneDataOffset);

          uint32_t stride = 0;
          switch (primvar->type)
          {
          case GiPrimvarType::Float:
          case GiPrimvarType::Int:
            stride = 1;
            break;
          case GiPrimvarType::Int2:
          case GiPrimvarType::Vec2:
            stride = 2;
            break;
          case GiPrimvarType::Int3:
          case GiPrimvarType::Vec3:
            stride = 3;
            break;
          case GiPrimvarType::Int4:
          case GiPrimvarType::Vec4:
            stride = 4;
            break;
          default:
            assert(false);
            GB_ERROR("coding error: unhandled type size!");
            continue;
          }
          stride -= 1; // [0, 3] range -> 2 bit

          assert(stride < 4);
          static_assert(int(GiPrimvarInterpolation::COUNT) <= 4, "Enum exceeds 2 bits");

          uint32_t info = ((uint32_t(sceneDataOffset) / rp::SCENE_DATA_ALIGNMENT) & rp::SCENE_DATA_OFFSET_MASK) |
                          (stride << rp::SCENE_DATA_STRIDE_OFFSET) |
                          (uint32_t(primvar->interpolation) << rp::SCENE_DATA_INTERPOLATION_OFFSET);
          preamble.sceneDataInfos[i] = info;
        }

        // Upload GPU data
        CgpuBlas blas;
        CgpuBuffer tmpPositionBuffer;
        CgpuBuffer tmpIndexBuffer;
        CgpuBuffer payloadBuffer;
        rp::BlasPayload payload;

        uint64_t tmpIndexBufferSize = indicesSize;
        uint64_t tmpPositionBufferSize = positionData.size() * sizeof(float);

        // Create data buffers
        if (!cgpuCreateBuffer(s_ctx, {
                                .usage = CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::TransferDst,
                                .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                                .size = payloadBufferSize,
                                .debugName = "BlasPayloadBuffer"
                              }, &payloadBuffer))
        {
          GB_ERROR("failed to allocate BLAS payload buffer memory");
          goto fail_cleanup;
        }

        if (!cgpuCreateBuffer(s_ctx, {
                                .usage = CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::AccelerationStructureBuild,
                                .memoryProperties = CgpuMemoryProperties::HostVisible,
                                .size = tmpPositionBufferSize,
                                .debugName = "BlasVertexPositionsTmp"
                              }, &tmpPositionBuffer))
        {
          GB_ERROR("failed to allocate BLAS temp vertex position memory");
          goto fail_cleanup;
        }

        if (!cgpuCreateBuffer(s_ctx, {
                                .usage = CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::AccelerationStructureBuild,
                                .memoryProperties = CgpuMemoryProperties::HostVisible,
                                .size = tmpIndexBufferSize,
                                .debugName = "BlasIndicesTmp"
                              }, &tmpIndexBuffer))
        {
          GB_ERROR("failed to allocate BLAS temp indices memory");
          goto fail_cleanup;
        }

        // Copy data to GPU
        {
          void* ptr = cgpuGetBufferCpuPtr(s_ctx, tmpPositionBuffer);
          memcpy(ptr, positionData.data(), tmpPositionBufferSize);
        }
        {
          void* ptr = cgpuGetBufferCpuPtr(s_ctx, tmpIndexBuffer);
          memcpy(ptr, indexData.data(), tmpIndexBufferSize);
        }

        if (!s_stager->stageToBuffer((uint8_t*) &preamble, preambleSize, payloadBuffer, 0) ||
            !s_stager->stageToBuffer((uint8_t*) indexData.data(), indicesSize, payloadBuffer, indexBufferOffset) ||
            !s_stager->stageToBuffer((uint8_t*) vertexData.data(), verticesSize, payloadBuffer, vertexBufferOffset) ||
            !s_stager->stageToBuffer((uint8_t*) faceIdData.data(), faceIdsSize, payloadBuffer, faceIdsBufferOffset))
        {
          GB_ERROR("failed to stage BLAS data");
          goto fail_cleanup;
        }

        for (size_t i = 0; i < primvars.size(); i++)
        {
          if (preamble.sceneDataInfos[i] == rp::SCENE_DATA_INVALID)
          {
            continue;
          }

          const GiPrimvarData* s = primvars[i];
          if (!s_stager->stageToBuffer(&s->data[0], s->data.size(), payloadBuffer, sceneDataOffsets[i]))
          {
            GB_ERROR("failed to stage BLAS primvar");
            goto fail_cleanup;
          }
        }

        s_stager->flush();

        // Build BLAS
        {
          const GiMaterial* material = shaderCache->materials[materialIndex];

          bool blasCreated = cgpuCreateBlas(s_ctx, {
                                              .vertexPosBuffer = tmpPositionBuffer,
                                              .indexBuffer = tmpIndexBuffer,
                                              .maxVertex = (uint32_t) positionData.size(),
                                              .triangleCount = (uint32_t) indexData.size() / 3,
                                              .isOpaque = !material->mcMat->hasCutoutTransparency,
                                              .debugName = mesh->name.c_str()
                                            }, &blas);

          if (!blasCreated)
          {
            GB_ERROR("failed to allocate BLAS vertex memory");
            goto fail_cleanup;
          }
        }

        cgpuDestroyBuffer(s_ctx, tmpPositionBuffer);
        tmpPositionBuffer.handle = 0;
        cgpuDestroyBuffer(s_ctx, tmpIndexBuffer);
        tmpIndexBuffer.handle = 0;

        // Append BLAS payload data
        {
          uint64_t payloadBufferAddress = cgpuGetBufferGpuAddress(s_ctx, payloadBuffer);
          if (payloadBufferAddress == 0)
          {
            GB_ERROR("failed to get index-vertex buffer address");
            goto fail_cleanup;
          }

          uint32_t bitfield = 0;
          if (mesh->flipFacing)
          {
            bitfield |= rp::BLAS_PAYLOAD_BITFLAG_FLIP_FACING;
          }
          if (mesh->doubleSided)
          {
            bitfield |= rp::BLAS_PAYLOAD_BITFLAG_DOUBLE_SIDED;
          }

          uint64_t vertexBufferSize = (vertexBufferOffset/* account for align */ - indexBufferOffset/* account for preamble */);
          payload = rp::BlasPayload{
            .bufferAddress = payloadBufferAddress,
            .vertexOffset = uint32_t(vertexBufferSize / sizeof(rp::FVertex)), // offset to skip index buffer
            .bitfield = bitfield
          };
        }

        mesh->gpuData = GiMeshGpuData{
          .blas = blas,
          .payloadBuffer = payloadBuffer,
          .payload = payload
        };

        // (we ignore padding and the preamble in the reporting, but they are negligible)
        totalVerticesSize += verticesSize;
        totalIndicesSize += indicesSize;

        if (false) // not executed in success case
        {
fail_cleanup:
          if (payloadBuffer.handle)
            cgpuDestroyBuffer(s_ctx, payloadBuffer);
          if (tmpPositionBuffer.handle)
            cgpuDestroyBuffer(s_ctx, tmpPositionBuffer);
          if (tmpIndexBuffer.handle)
            cgpuDestroyBuffer(s_ctx, tmpIndexBuffer);
          if (blas.handle)
            cgpuDestroyBlas(s_ctx, blas);

          continue;
        }
      }

      const auto& data = mesh->gpuData;
      if (!data.has_value())
      {
        continue; // invalid geometry or an error occurred
      }

      totalIndicesSize += mesh->cpuData.faceCount * sizeof(uint32_t) * 3;
      totalVerticesSize += mesh->cpuData.vertexCount * sizeof(rp::FVertex);

      for (size_t i = 0; i < mesh->instanceTransforms.size(); i++)
      {
        // Create BLAS instance for TLAS.
        glm::mat3x4 transform = glm::mat3x4(glm::mat4(mesh->transform) * glm::mat4(mesh->instanceTransforms[i]));

        CgpuBlasInstance blasInstance;
        blasInstance.as = data->blas;
        blasInstance.hitGroupIndex = materialIndex * 2; // always two hit groups per material: regular & shadow
        blasInstance.instanceCustomIndex = uint32_t(blasPayloads.size());
        memcpy(blasInstance.transform, glm::value_ptr(transform), sizeof(float) * 12);

        blasInstances.push_back(blasInstance);
        blasPayloads.push_back(data->payload);
        instanceIds.push_back(mesh->instanceIds[i]);
      }
    }
  }

  GiBvh* _giCreateBvh(GiScene* scene, const GiShaderCache* shaderCache)
  {
    GiBvh* bvh = nullptr;

    GB_LOG("creating bvh..");
    fflush(stdout);

    // Build BLASes.
    CgpuTlas tlas;
    std::vector<CgpuBlasInstance> blasInstances;
    std::vector<rp::BlasPayload> blasPayloads;
    std::vector<int> instanceIds;
    uint64_t indicesSize = 0;
    uint64_t verticesSize = 0;
    CgpuBuffer blasPayloadsBuffer;
    CgpuBuffer instanceIdsBuffer;

    _giBuildGeometryStructures(scene, shaderCache, blasInstances, blasPayloads, instanceIds, indicesSize, verticesSize);

    GB_LOG("BLAS builds finished");
    GB_LOG("> {} unique BLAS", blasPayloads.size());
    GB_LOG("> {} BLAS instances", blasInstances.size());
    GB_LOG("> {:.2f} MiB total indices", indicesSize * BYTES_TO_MIB);
    GB_LOG("> {:.2f} MiB total vertices", verticesSize * BYTES_TO_MIB);

    // Create TLAS.
    {
      if (!cgpuCreateTlas(s_ctx, {
                            .instanceCount = (uint32_t) blasInstances.size(),
                            .instances = blasInstances.data()
                          }, &tlas))
      {
        GB_ERROR("failed to create TLAS");
        goto cleanup;
      }

      GB_LOG("TLAS build finished");
    }

    // Upload blas buffer addresses to GPU.
    {
      uint64_t bufferSize = (blasPayloads.empty() ? 1 : blasPayloads.size()) * sizeof(rp::BlasPayload);

      if (!cgpuCreateBuffer(s_ctx, {
                              .usage = CgpuBufferUsage::Storage | CgpuBufferUsage::TransferDst,
                              .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                              .size = bufferSize,
                              .debugName = "BlasPayloadAddresses"
                            }, &blasPayloadsBuffer))
      {
        GB_ERROR("failed to create BLAS payloads buffer");
        goto cleanup;
      }

      if (!blasPayloads.empty() && !s_stager->stageToBuffer((uint8_t*) blasPayloads.data(), bufferSize, blasPayloadsBuffer))
      {
        GB_ERROR("failed to upload addresses to BLAS payload buffer");
        goto cleanup;
      }
    }

    // Upload instance IDs.
    {
      uint64_t bufferSize = (instanceIds.empty() ? 1 : instanceIds.size()) * sizeof(int);

      if (!cgpuCreateBuffer(s_ctx, {
                              .usage = CgpuBufferUsage::Storage | CgpuBufferUsage::TransferDst,
                              .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                              .size = bufferSize,
                              .debugName = "InstanceIds"
                            }, &instanceIdsBuffer))
      {
        GB_ERROR("failed to create instance IDs buffer");
        goto cleanup;
      }

      if (!instanceIds.empty() && !s_stager->stageToBuffer((uint8_t*) instanceIds.data(), bufferSize, instanceIdsBuffer))
      {
        GB_ERROR("failed to upload instance IDs");
        goto cleanup;
      }
    }

    // Fill cache struct.
    bvh = new GiBvh;
    bvh->blasPayloadsBuffer = blasPayloadsBuffer;
    bvh->instanceIdsBuffer = instanceIdsBuffer;
    bvh->scene = scene;
    bvh->tlas = tlas;

cleanup:
    if (!bvh)
    {
      assert(false);
      if (blasPayloadsBuffer.handle)
      {
        cgpuDestroyBuffer(s_ctx, blasPayloadsBuffer);
      }
      if (instanceIdsBuffer.handle)
      {
        cgpuDestroyBuffer(s_ctx, instanceIdsBuffer);
      }
      if (tlas.handle)
      {
        cgpuDestroyTlas(s_ctx, tlas);
      }
    }

    return bvh;
  }

  void _giDestroyBvh(GiBvh* bvh)
  {
    cgpuDestroyTlas(s_ctx, bvh->tlas);
    cgpuDestroyBuffer(s_ctx, bvh->blasPayloadsBuffer);
    cgpuDestroyBuffer(s_ctx, bvh->instanceIdsBuffer);
    delete bvh;
  }

  GiShaderCache* _giCreateShaderCache(const GiRenderParams& params)
  {
    struct HitShaderCompInfo
    {
      std::vector<uint8_t> spv;
      std::vector<uint8_t> shadowSpv;
    };
    struct HitGroupCompInfo
    {
      GiMaterial* material = nullptr;
      GiGlslShaderGen::MaterialGenInfo genInfo;

      HitShaderCompInfo closestHitInfo;
      std::optional<HitShaderCompInfo> anyHitInfo;
    };

    const GiRenderSettings& renderSettings = params.renderSettings;
    GiScene* scene = params.scene;
    OffsetAllocator::Allocator& texAllocator = scene->texAllocator;

    uint32_t aovMask = 0;
    for (const GiAovBinding& binding : params.aovBindings)
    {
      if (binding.aovId == GiAovId::ClockCycles && !s_ctxFeatures.shaderClock)
      {
        GB_ERROR("clock cycles AOV misses device feature - ignoring");
        continue;
      }

      aovMask |= (1 << int(binding.aovId));
    }

    std::set<GiMaterial*> materialSet;
    for (auto* m : scene->meshes)
    {
      if (m->material)
      {
        materialSet.insert(m->material);
      }
    }

    std::vector<GiMaterial*> materials(materialSet.begin(), materialSet.end());

    GB_LOG("material count: {}", materials.size());
    GB_LOG("creating shader cache..");
    fflush(stdout);

    GiShaderCache* cache = nullptr;
    CgpuPipeline pipeline;
    std::array<CgpuBindSet, 3> bindSets;
    CgpuShader rgenShader;
    std::vector<GiMaterialGpuData> newMaterialGpuDatas;
    std::vector<CgpuRtHitGroup> hitGroups;
    std::vector<CgpuShader> missShaders;
    std::vector<McTextureDescription> textureDescriptions;
    std::vector<GiImageBinding> imageBindings;
    std::vector<HitGroupCompInfo> hitGroupCompInfos;
    std::vector<const GiMaterial*> cachedMaterials;

    uint32_t maxRayPayloadSize = _GetRpMainMaxRayPayloadSize(renderSettings.mediumStackSize);
    uint32_t maxRayHitAttributeSize = _GetRpMainMaxRayHitAttributeSize();
    uint32_t maxTextureIndex = 0;

    GiGlslShaderGen::CommonShaderParams commonParams = {
      .aovMask = aovMask,
#ifndef NDEBUG
      .debugPrintf = s_ctxFeatures.debugPrintf,
#else
      .debugPrintf = false,
#endif
      .mediumStackSize = renderSettings.mediumStackSize,
      .progressiveAccumulation = renderSettings.progressiveAccumulation
    };

    bool needsAuxOutput = bool(aovMask & (1 << int(GiAovId::Albedo)));
    s_shaderGen->setAuxiliaryOutputEnabled(needsAuxOutput);

    // Create per-material hit shaders.
    {
      // 1. Generate GLSL from MDL in parallel
      std::mutex cachedMaterialsMutex;
      std::mutex hitGroupCompInfoMutex;
      hitGroupCompInfos.reserve(materials.size());

      std::atomic_bool threadWorkFailed = false;
#pragma omp parallel for
      for (int i = 0; i < int(materials.size()); i++)
      {
        GiMaterial* material = materials[i];
        if (material->gpuData)
        {
          std::lock_guard guard(cachedMaterialsMutex);
          cachedMaterials.push_back(material);
          continue;
        }

        GiGlslShaderGen::MaterialGenInfo genInfo;
        if (!s_shaderGen->generateMaterialInfo(*material->mcMat, genInfo))
        {
          GB_ERROR("failed to generate code for material {}", material->name);
          threadWorkFailed = true;
          continue;
        }

        HitGroupCompInfo groupInfo;
        groupInfo.material = material;
        groupInfo.genInfo = genInfo;

        if (material->mcMat->hasCutoutTransparency)
        {
          groupInfo.anyHitInfo = HitShaderCompInfo{};
        }

        std::lock_guard guard(hitGroupCompInfoMutex);
        hitGroupCompInfos.push_back(groupInfo);
      }
      if (threadWorkFailed)
      {
        goto cleanup;
      }

      // 2. Allocate texture indices for new shaders & collect textures to load.
      for (HitGroupCompInfo& groupInfo : hitGroupCompInfos)
      {
        GiMaterial* material = groupInfo.material;
        assert(material);

        const GiGlslShaderGen::MaterialGenInfo& genInfo = groupInfo.genInfo;
        for (const McTextureDescription& tr : genInfo.textureDescriptions)
        {
          textureDescriptions.push_back(tr);
        }
      }

      // 3. Upload textures and assign images to new material GPU data.
      std::vector<GiImagePtr> images;
      if (textureDescriptions.size() > 0 && !s_texSys->loadTextureDescriptions(textureDescriptions, images))
      {
        goto cleanup;
      }

      uint32_t imageCounter = 0;
      newMaterialGpuDatas.reserve(hitGroupCompInfos.size());
      for (HitGroupCompInfo& groupInfo : hitGroupCompInfos)
      {
        GiMaterial* material = groupInfo.material;

        const GiGlslShaderGen::MaterialGenInfo& genInfo = groupInfo.genInfo;
        auto texCount = uint32_t(genInfo.textureDescriptions.size());

        GiMaterialGpuData gpuData;
        if (texCount > 0)
        {
          gpuData.texOffsetAllocation = texAllocator.allocate(texCount);

          if (gpuData.texOffsetAllocation.offset == OffsetAllocator::Allocation::NO_SPACE)
          {
            GI_FATAL("max number of textures exceeded");
          }
        }
        for (const McTextureDescription& tr : genInfo.textureDescriptions)
        {
          gpuData.images.push_back(images[imageCounter++]);
        }
        newMaterialGpuDatas.push_back(gpuData);
      }

      // 4. Generate final hit shader GLSL sources.
#pragma omp parallel for
      for (int i = 0; i < int(hitGroupCompInfos.size()); i++)
      {
        HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

        GiMaterial* material = compInfo.material;
        const McMaterial* mcMat = material->mcMat;

        uint32_t texOffset = newMaterialGpuDatas[i].texOffsetAllocation.offset;
        auto sceneDataCount = uint32_t(mcMat->sceneDataNames.size())
          - int(bool(mcMat->cameraPositionSceneDataIndex));

        // Closest hit
        {
          GiGlslShaderGen::ClosestHitShaderParams hitParams = {
            .baseFileName = "rp_main.chit",
            .commonParams = commonParams,
            .directionalBias = mcMat->directionalBias,
            .enableSceneTransforms = mcMat->requiresSceneTransforms,
            .cameraPositionSceneDataIndex = mcMat->cameraPositionSceneDataIndex,
            .hasBackfaceBsdf = mcMat->hasBackfaceBsdf,
            .hasBackfaceEdf = mcMat->hasBackfaceEdf,
            .hasCutoutTransparency = mcMat->hasCutoutTransparency,
            .hasVolumeAbsorptionCoeff = mcMat->hasVolumeAbsorptionCoeff,
            .hasVolumeScatteringCoeff = mcMat->hasVolumeScatteringCoeff,
            .isEmissive = mcMat->isEmissive,
            .isThinWalled = mcMat->isThinWalled,
            .nextEventEstimation = renderSettings.nextEventEstimation,
            .sceneDataCount = sceneDataCount,
            .shadingGlsl = compInfo.genInfo.glslSource,
            .textureIndexOffset = texOffset
          };

          if (!s_shaderGen->generateClosestHitSpirv(hitParams, compInfo.closestHitInfo.spv))
          {
            threadWorkFailed = true;
            continue;
          }
        }

        // Any hit
        if (compInfo.anyHitInfo)
        {
          GiGlslShaderGen::AnyHitShaderParams hitParams = {
            .baseFileName = "rp_main.ahit",
            .commonParams = commonParams,
            .enableSceneTransforms = mcMat->requiresSceneTransforms,
            .cameraPositionSceneDataIndex = mcMat->cameraPositionSceneDataIndex,
            .opacityEvalGlsl = compInfo.genInfo.glslSource,
            .sceneDataCount = sceneDataCount,
            .textureIndexOffset = texOffset
          };

          hitParams.shadowTest = false;
          if (!s_shaderGen->generateAnyHitSpirv(hitParams, compInfo.anyHitInfo->spv))
          {
            threadWorkFailed = true;
            continue;
          }

          hitParams.shadowTest = true;
          if (!s_shaderGen->generateAnyHitSpirv(hitParams, compInfo.anyHitInfo->shadowSpv))
          {
            threadWorkFailed = true;
            continue;
          }
        }
      }
      if (threadWorkFailed)
      {
        goto cleanup;
      }

      // 5. Compile & assign back the shaders.
      auto hitGroupCount = (uint32_t) hitGroupCompInfos.size();

      std::vector<CgpuShaderCreateInfo> createInfos;
      createInfos.reserve(hitGroupCount * 2);

      for (uint32_t i = 0; i < hitGroupCount; i++)
      {
        const HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

        // regular hit group
        {
          const std::vector<uint8_t>& cSpv = compInfo.closestHitInfo.spv;
          createInfos.push_back({ .size = cSpv.size(), .source = cSpv.data(), .stageFlags = CgpuShaderStage::ClosestHit,
                                  .maxRayPayloadSize = maxRayPayloadSize, .maxRayHitAttributeSize = maxRayHitAttributeSize, });

          if (compInfo.anyHitInfo)
          {
            const std::vector<uint8_t>& aSpv = compInfo.anyHitInfo->spv;
            createInfos.push_back({ .size = aSpv.size(), .source = aSpv.data(), .stageFlags = CgpuShaderStage::AnyHit,
                                    .maxRayPayloadSize = maxRayPayloadSize, .maxRayHitAttributeSize = maxRayHitAttributeSize, });
          }
        }

        // shadow hit group
        if (compInfo.anyHitInfo)
        {
          const std::vector<uint8_t>& aSpv = compInfo.anyHitInfo->shadowSpv;
          createInfos.push_back({ .size = aSpv.size(), .source = aSpv.data(), .stageFlags = CgpuShaderStage::AnyHit,
                                  .maxRayPayloadSize = maxRayPayloadSize, .maxRayHitAttributeSize = maxRayHitAttributeSize, });
        }
      }

      std::vector<CgpuShader> hitShaders(createInfos.size());
      if (!cgpuCreateShadersParallel(s_ctx, (uint32_t) createInfos.size(), createInfos.data(), hitShaders.data()))
      {
        goto cleanup;
      }

      for (uint32_t i = 0, hitShaderCounter = 0; i < hitGroupCount; i++)
      {
        const HitGroupCompInfo& compInfo = hitGroupCompInfos[i];
        GiMaterialGpuData& gpuData = newMaterialGpuDatas[i];

        // regular hit group
        {
          gpuData.closestHit = hitShaders[hitShaderCounter++];

          if (compInfo.anyHitInfo)
          {
            gpuData.anyHits.push_back(hitShaders[hitShaderCounter++]);
          }
        }

        // shadow hit group
        if (compInfo.anyHitInfo)
        {
            gpuData.anyHits.push_back(hitShaders[hitShaderCounter++]);
        }
      }
    }

    // Collect hit shader groups.
    hitGroups.reserve(materials.size());
    {
      std::unordered_map<const GiMaterial*, const GiMaterialGpuData*> gpuDatas; // material indices are important for SBT addressing

      for (const GiMaterial* material : cachedMaterials)
      {
        gpuDatas[material] = &(*material->gpuData);
      }
      for (size_t i = 0; i < hitGroupCompInfos.size(); i++)
      {
        const HitGroupCompInfo& compInfo = hitGroupCompInfos[i];
        GiMaterialGpuData& gpuData = newMaterialGpuDatas[i];
        gpuDatas[compInfo.material] = &gpuData;
      }

      for (size_t i = 0; i < materials.size(); i++)
      {
        const GiMaterialGpuData& gpuData = *gpuDatas[materials[i]];

        // regular hit group
        CgpuRtHitGroup hitGroup;
        hitGroup.closestHitShader = gpuData.closestHit;
        if (!gpuData.anyHits.empty())
        {
          hitGroup.anyHitShader = gpuData.anyHits[0];
        }
        hitGroups.push_back(hitGroup);

        // shadow hit group
        CgpuRtHitGroup shadowHitGroup;
        if (!gpuData.anyHits.empty())
        {
          shadowHitGroup.anyHitShader = gpuData.anyHits[1];
        }
        hitGroups.push_back(shadowHitGroup);
      };
    }

    // Create ray generation shader.
    {
      GiGlslShaderGen::RaygenShaderParams rgenParams = {
        .clippingPlanes = renderSettings.clippingPlanes,
        .commonParams = commonParams,
        .depthOfField = renderSettings.depthOfField,
        .filterImportanceSampling = renderSettings.filterImportanceSampling,
        .jitteredSampling = renderSettings.jitteredSampling,
        .materialCount = uint32_t(materials.size()),
        .nextEventEstimation = renderSettings.nextEventEstimation,
        .reorderInvocations = s_ctxFeatures.rayTracingInvocationReorder
      };

      std::vector<uint8_t> spv;
      if (!s_shaderGen->generateRgenSpirv("rp_main.rgen", rgenParams, spv))
      {
        goto cleanup;
      }

      if (!cgpuCreateShader(s_ctx, {
                              .size = spv.size(),
                              .source = spv.data(),
                              .stageFlags = CgpuShaderStage::RayGen,
                              .maxRayPayloadSize = maxRayPayloadSize,
                              .maxRayHitAttributeSize = maxRayHitAttributeSize
                            }, &rgenShader))
      {
        goto cleanup;
      }
    }

    // Create miss shaders.
    {
      GiGlslShaderGen::MissShaderParams missParams = {
        .commonParams = commonParams,
        .domeLightCameraVisible = renderSettings.domeLightCameraVisible
      };

      // regular miss shader
      {
        std::vector<uint8_t> spv;
        if (!s_shaderGen->generateMissSpirv("rp_main.miss", missParams, spv))
        {
          goto cleanup;
        }

        CgpuShader missShader;
        if (!cgpuCreateShader(s_ctx, {
                                .size = spv.size(),
                                .source = spv.data(),
                                .stageFlags = CgpuShaderStage::Miss,
                                .maxRayPayloadSize = maxRayPayloadSize,
                                .maxRayHitAttributeSize = maxRayHitAttributeSize
                              }, &missShader))
        {
          goto cleanup;
        }

        missShaders.push_back(missShader);
      }

      // shadow test miss shader
      {
        std::vector<uint8_t> spv;
        if (!s_shaderGen->generateMissSpirv("rp_main_shadow.miss", missParams, spv))
        {
          goto cleanup;
        }

        CgpuShader missShader;
        if (!cgpuCreateShader(s_ctx, {
                                .size = spv.size(),
                                .source = spv.data(),
                                .stageFlags = CgpuShaderStage::Miss,
                                .maxRayPayloadSize = maxRayPayloadSize,
                                .maxRayHitAttributeSize = maxRayHitAttributeSize
                              }, &missShader))
        {
          goto cleanup;
        }

        missShaders.push_back(missShader);
      }
    }

    // Create RT pipeline.
    {
      GB_LOG("creating RT pipeline..");
      fflush(stdout);

      cgpuCreateRtPipeline(s_ctx, {
        .rgenShader = rgenShader,
        .missShaderCount = (uint32_t)missShaders.size(),
        .missShaders = missShaders.data(),
        .hitGroupCount = (uint32_t)hitGroups.size(),
        .hitGroups = hitGroups.data(),
        .maxRayPayloadSize = maxRayPayloadSize,
        .maxRayHitAttributeSize = maxRayHitAttributeSize,
        .payloadStride = 2
      }, &pipeline);

      cgpuCreateBindSets(s_ctx, pipeline, bindSets.data(), (uint32_t) bindSets.size());
    }

    // Assign GPU data to materials.
    for (size_t i = 0; i < hitGroupCompInfos.size(); i++)
    {
      HitGroupCompInfo& compInfo = hitGroupCompInfos[i];
      compInfo.material->gpuData = newMaterialGpuDatas[i];
    }

    // Collect image bindings.
    for (size_t i = 0, j = 0; i < materials.size(); i++)
    {
      const GiMaterial* material = materials[i];
      const GiMaterialGpuData& gpuData = *material->gpuData;
      uint32_t texOffset = gpuData.texOffsetAllocation.offset;

      uint32_t texCount = 0;
      for (GiImagePtr img : gpuData.images)
      {
        uint32_t index = texOffset + texCount;
        imageBindings.push_back({ .image = img, .index = index });
        if (index > maxTextureIndex)
        {
          maxTextureIndex = index;
        }
        texCount++;
      }
    }

    cache = new GiShaderCache;
    cache->aovMask = aovMask;
    cache->bindSets = bindSets;
    cache->domeLightCameraVisible = renderSettings.domeLightCameraVisible;
    cache->imageBindings = std::move(imageBindings);
    cache->maxTextureIndex = maxTextureIndex;
    cache->materials.resize(materials.size());
    for (uint32_t i = 0; i < cache->materials.size(); i++)
    {
      cache->materials[i] = materials[i];
    }
    cache->missShaders = missShaders;
    cache->pipeline = pipeline;
    cache->rgenShader = rgenShader;
    cache->scene = scene;

cleanup:
    // Note: the reason why we properly destruct resources instead of hard exit on shader
    // compilation errors is that we want shader hotloading to not bloat resource usage.
    if (!cache)
    {
      if (rgenShader.handle)
      {
        cgpuDestroyShader(s_ctx, rgenShader);
      }
      for (CgpuShader shader : missShaders)
      {
        cgpuDestroyShader(s_ctx, shader);
      }
      if (pipeline.handle)
      {
        cgpuDestroyBindSets(s_ctx, bindSets.data(), (uint32_t) bindSets.size());
        cgpuDestroyPipeline(s_ctx, pipeline);
      }
      for (GiMaterialGpuData& gpuData : newMaterialGpuDatas)
      {
        giDestroyMaterialGpuData(scene, gpuData);
      }
    }
    return cache;
  }

  void _giDestroyShaderCache(GiShaderCache* cache)
  {
    GiScene* scene = cache->scene;

    cgpuDestroyShader(s_ctx, cache->rgenShader);
    for (CgpuShader shader : cache->missShaders)
    {
      cgpuDestroyShader(s_ctx, shader);
    }
    cgpuDestroyBindSets(s_ctx, cache->bindSets.data(), (uint32_t) cache->bindSets.size());
    cgpuDestroyPipeline(s_ctx, cache->pipeline);
    delete cache;
  }

  GiSceneDirtyFlags _CalcDirtyFlagsForRenderParams(const GiRenderParams& a/*new*/,
                                                   const GiRenderParams& b/*old*/)
  {
    GiSceneDirtyFlags flags = {};

    if (memcmp(&a, &b, sizeof(GiRenderParams)) == 0)
    {
      return flags;
    }

    bool aovCountChanged = a.aovBindings.size() != b.aovBindings.size();
    bool aovsChanged = aovCountChanged;
    bool aovDefaultsChanged = false;
    bool renderBufferChanged = false;
    if (!aovCountChanged)
    {
      auto& oldAovs = b.aovBindings;
      auto& newAovs = a.aovBindings;

      for (int i = 0; i < newAovs.size(); i++)
      {
        auto& oldAov = oldAovs[i];
        auto& newAov = newAovs[i];

        if (oldAov.aovId != newAov.aovId)
        {
          aovsChanged = true;
        }
        if (oldAov.renderBuffer != newAov.renderBuffer)
        {
          renderBufferChanged = true;
        }
        if (memcmp(oldAov.clearValue, newAov.clearValue, GI_MAX_AOV_COMP_SIZE) != 0)
        {
          aovDefaultsChanged = true;
        }
      }
    }
    if (aovsChanged)
    {
      flags |= GiSceneDirtyFlags::DirtyShadersAll | GiSceneDirtyFlags::DirtyBindSets;
    }
    if (aovsChanged || aovDefaultsChanged)
    {
      flags |= GiSceneDirtyFlags::DirtyAovBindingDefaults | GiSceneDirtyFlags::DirtyFramebuffer;
    }
    if (renderBufferChanged)
    {
      flags |= GiSceneDirtyFlags::DirtyFramebuffer | GiSceneDirtyFlags::DirtyBindSets;
    }

    if (memcmp(&a.camera, &b.camera, sizeof(GiCameraDesc)) != 0 ||
        memcmp(&a.renderSettings, &b.renderSettings, sizeof(GiRenderSettings)) != 0)
    {
      flags |= GiSceneDirtyFlags::DirtyFramebuffer;
    }

    if (a.domeLight != b.domeLight ||
        a.scene != b.scene ||
        a.scene->domeLightTexture != b.scene->domeLightTexture)
    {
      flags |= GiSceneDirtyFlags::DirtyFramebuffer | GiSceneDirtyFlags::DirtyBindSets;
    }

    const GiRenderSettings& ra = a.renderSettings;
    const GiRenderSettings& rb = b.renderSettings;

    if (ra.domeLightCameraVisible != rb.domeLightCameraVisible)
    {
      flags |= GiSceneDirtyFlags::DirtyShadersMiss;
    }

    if (ra.clippingPlanes != rb.clippingPlanes ||
        ra.depthOfField != rb.depthOfField ||
        ra.filterImportanceSampling != rb.filterImportanceSampling ||
        ra.jitteredSampling != rb.jitteredSampling ||
        ra.maxVolumeWalkLength != rb.maxVolumeWalkLength)
    {
      flags |= GiSceneDirtyFlags::DirtyShadersRgen;
    }

    if (ra.mediumStackSize != rb.mediumStackSize ||
        ra.nextEventEstimation != rb.nextEventEstimation ||
        ra.progressiveAccumulation != rb.progressiveAccumulation)
    {
      flags |= GiSceneDirtyFlags::DirtyShadersAll;
    }

    return flags;
  }

  GiStatus giRender(const GiRenderParams& params)
  {
    s_stager->flush();

    GiScene* scene = params.scene;
    const GiRenderSettings& renderSettings = params.renderSettings;

    // Allocate memory for render buffers
    for (const GiAovBinding& binding : params.aovBindings)
    {
      GiRenderBuffer* renderBuffer = binding.renderBuffer;

      if (renderBuffer->deviceMem.handle)
      {
        continue;
      }

      if (!cgpuCreateBuffer(s_ctx, {
                              .usage = CgpuBufferUsage::Storage | CgpuBufferUsage::TransferSrc,
                              .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                              .size = renderBuffer->size,
                              .debugName = "RenderBufferGpu"
                            }, &renderBuffer->deviceMem))
      {
        GB_ERROR("failed to allocate render buffer");
        return GiStatus::Error;
      }

      if (!cgpuCreateBuffer(s_ctx, {
                              .usage = CgpuBufferUsage::TransferDst,
                              .memoryProperties = CgpuMemoryProperties::HostVisible |
                                                  CgpuMemoryProperties::HostCached,
                              .size = renderBuffer->size,
                              .debugName = "RenderBufferCpu"
                            }, &renderBuffer->hostMem))
      {
        GB_ERROR("failed to allocate render buffer");
        cgpuDestroyBuffer(s_ctx, renderBuffer->deviceMem);
        return GiStatus::Error;
      }

      renderBuffer->mappedHostMem = cgpuGetBufferCpuPtr(s_ctx, renderBuffer->hostMem);

      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBindSets;
      s_resetSampleOffset = true;
    }

    if (s_forceShaderCacheInvalid)
    {
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyShadersAll | GiSceneDirtyFlags::DirtyFramebuffer;
      s_forceShaderCacheInvalid = false;
    }

    if (s_resetSampleOffset)
    {
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
      s_resetSampleOffset = false;
    }

    if (GiSceneDirtyFlags flags = _CalcDirtyFlagsForRenderParams(params, scene->oldRenderParams); bool(flags))
    {
      scene->dirtyFlags |= flags;
      scene->oldRenderParams = params;
    }

    if (bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyShadersHit))
    {
      for (GiMaterial* mat : scene->materials)
      {
        if (!mat->gpuData)
        {
          continue;
        }

        GiMaterialGpuData& gpuData = *mat->gpuData;

        if (gpuData.texOffsetAllocation.offset != OffsetAllocator::Allocation::NO_SPACE)
        {
          scene->texAllocator.free(gpuData.texOffsetAllocation);
        }
        if (gpuData.closestHit.handle)
        {
          cgpuDestroyShader(s_ctx, gpuData.closestHit);
        }
        for (CgpuShader shader : gpuData.anyHits)
        {
          cgpuDestroyShader(s_ctx, shader);
        }

        mat->gpuData.reset();
      }

      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyPipeline;
    }

    // TODO: individual rgen & miss shader recompilation
    if (!scene->shaderCache ||
        bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyPipeline) ||
        bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyShadersRgen) ||
        bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyShadersHit) ||
        bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyShadersMiss))
    {
      GiShaderCache* oldShaderCache = scene->shaderCache;

      scene->shaderCache = _giCreateShaderCache(params);

      if (oldShaderCache)
      {
        // Delay destruction to reuse images
        _giDestroyShaderCache(oldShaderCache);
      }

      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyShadersRgen;
      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyShadersHit;
      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyShadersMiss;
      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyPipeline;
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer |
                           GiSceneDirtyFlags::DirtyBvh/*SBT*/ |
                           GiSceneDirtyFlags::DirtyBindSets;
    }

    if (!scene->shaderCache)
    {
      return GiStatus::Error;
    }

    if (!scene->bvh || bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyBvh))
    {
      if (scene->bvh) _giDestroyBvh(scene->bvh);

      scene->bvh = _giCreateBvh(scene, scene->shaderCache);

      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyBvh;
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer | GiSceneDirtyFlags::DirtyBindSets;
    }

    if (bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyFramebuffer))
    {
      scene->sampleOffset = 0;
      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyFramebuffer;
    }

    if (bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyAovBindingDefaults))
    {
      if (scene->aovDefaultValues.handle)
      {
        s_deleteQueue->pushBack(scene->aovDefaultValues);
      }

      size_t conservativeSize = int(GiAovId::COUNT) * GI_MAX_AOV_COMP_SIZE;
      if (!cgpuCreateBuffer(s_ctx, {
                              .usage = CgpuBufferUsage::Storage | CgpuBufferUsage::TransferDst,
                              .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                              .size = conservativeSize,
                              .debugName = "AovDefaults"
                            }, &scene->aovDefaultValues))
      {
        GB_ERROR("failed to create AOV default values buffer");
        return GiStatus::Error;
      }

#ifdef GTL_VERBOSE
      std::string aovDebugMsg = "AOVs: ";
      for (uint32_t i = 0; i < params.aovBindings.size(); i++)
      {
        const GiAovBinding& binding = params.aovBindings[i];

        if (i != 0) aovDebugMsg += ", ";
        aovDebugMsg += GB_FMT("{}", int(binding.aovId));
      }
      GB_DEBUG("{}", aovDebugMsg);
#endif

      std::vector<uint8_t> defaultsData(conservativeSize);
      for (uint32_t i = 0; i < params.aovBindings.size(); i++)
      {
        const GiAovBinding& binding = params.aovBindings[i];
        memcpy(&defaultsData[int(binding.aovId) * GI_MAX_AOV_COMP_SIZE], &binding.clearValue[0], GI_MAX_AOV_COMP_SIZE);
      }

      if (!s_stager->stageToBuffer(&defaultsData[0], defaultsData.size(), scene->aovDefaultValues) ||
          !s_stager->flush())
      {
        GB_ERROR("failed to stage AOV default values");
        return GiStatus::Error;
      }

      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyAovBindingDefaults;
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBindSets;
    }

    const GiShaderCache* shaderCache = scene->shaderCache;
    const GiBvh* bvh = scene->bvh;

    // Upload dome lights
    glm::vec4 backgroundColor(0.0f);
    for (const GiAovBinding& binding : params.aovBindings)
    {
      if (binding.aovId != GiAovId::Color)
      {
        continue;
      }
      memcpy(&backgroundColor[0], binding.clearValue, GI_MAX_AOV_COMP_SIZE);
    }

    if (backgroundColor != scene->backgroundColor)
    {
      glm::u8vec4 u8BgColor(backgroundColor * 255.0f);
      s_stager->stageToImage(glm::value_ptr(u8BgColor), 4, scene->fallbackDomeLightTexture, 1, 1);
      scene->backgroundColor = backgroundColor;
    }

    if (scene->domeLight != params.domeLight)
    {
      if (scene->domeLightTexture &&
          scene->domeLightTexture->handle != scene->fallbackDomeLightTexture.handle)
      {
        // TODO: if we have multiple frames in flight, we need to wait for last frame's semaphore here
        scene->domeLightTexture.reset(); // frees memory immediately
        scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBindSets;
      }
      scene->domeLight = nullptr;

      GiDomeLight* domeLight = params.domeLight;
      if (domeLight)
      {
        const char* filePath = domeLight->textureFilePath.c_str();

        bool destroyImmediately = true;
        bool keepHdr = true;
        scene->domeLightTexture = s_texSys->loadTextureFromFilePath(filePath, destroyImmediately, keepHdr);

        if (!scene->domeLightTexture)
        {
          GB_ERROR("unable to load dome light texture at {}", filePath);
        }
        else
        {
          scene->domeLight = domeLight;
          scene->dirtyFlags |= GiSceneDirtyFlags::DirtyBindSets;
        }
      }
    }
    if (!scene->domeLight)
    {
      // Use fallback texture in case no dome light is set. We still have an explicit binding
      // for the fallback texture because we need the background color in case the textured
      // dome light is not supposed to be seen by the camera ('domeLightCameraVisible' option).
      scene->domeLightTexture = std::make_shared<CgpuImage>(scene->fallbackDomeLightTexture);
    }

    // Init state for goto error handling
    GiStatus result = GiStatus::Error;

    if (!scene->sphereLights.commitChanges())
    {
      GB_ERROR("{}:{}: light commit failed!", __FILE__, __LINE__);
    }
    if (!scene->distantLights.commitChanges())
    {
      GB_ERROR("{}:{}: light commit failed!", __FILE__, __LINE__);
    }
    if (!scene->rectLights.commitChanges())
    {
      GB_ERROR("{}:{}: light commit failed!", __FILE__, __LINE__);
    }
    if (!scene->diskLights.commitChanges())
    {
      GB_ERROR("{}:{}: light commit failed!", __FILE__, __LINE__);
    }

    if (!s_stager->flush())
    {
      GB_ERROR("{}:{}: stager flush failed!", __FILE__, __LINE__);
    }

    // FIXME: use values from Hydra camera and ensure they match the render buffers
    uint32_t imageWidth = params.aovBindings[0].renderBuffer->width;
    uint32_t imageHeight = params.aovBindings[0].renderBuffer->height;

    // Start command buffer.
    CgpuCommandBuffer commandBuffer;
    CgpuSemaphore semaphore;
    CgpuSignalSemaphoreInfo signalSemaphoreInfo;
    CgpuWaitSemaphoreInfo waitSemaphoreInfo;

    if (!cgpuCreateCommandBuffer(s_ctx, &commandBuffer))
      goto cleanup;

    if (!cgpuBeginCommandBuffer(s_ctx, commandBuffer))
      goto cleanup;

    // Update descriptor sets if needed
    if (bool(scene->dirtyFlags & GiSceneDirtyFlags::DirtyBindSets))
    {
      GB_DEBUG("updating descriptor sets");

      std::vector<CgpuBufferBinding> buffers;
      buffers.reserve(16);

      buffers.push_back({ .binding = rp::BINDING_INDEX_UNIFORM_DATA, .buffer = s_bumpAlloc->getBuffer(), .size = sizeof(rp::UniformData) });

      buffers.push_back({ .binding = rp::BINDING_INDEX_SPHERE_LIGHTS, .buffer = scene->sphereLights.buffer() });
      buffers.push_back({ .binding = rp::BINDING_INDEX_DISTANT_LIGHTS, .buffer = scene->distantLights.buffer() });
      buffers.push_back({ .binding = rp::BINDING_INDEX_RECT_LIGHTS, .buffer = scene->rectLights.buffer() });
      buffers.push_back({ .binding = rp::BINDING_INDEX_DISK_LIGHTS, .buffer = scene->diskLights.buffer() });
      buffers.push_back({ .binding = rp::BINDING_INDEX_BLAS_PAYLOADS, .buffer = bvh->blasPayloadsBuffer });
      buffers.push_back({ .binding = rp::BINDING_INDEX_INSTANCE_IDS, .buffer = bvh->instanceIdsBuffer });

      buffers.push_back({ .binding = rp::BINDING_INDEX_AOV_CLEAR_VALUES_F, .buffer = scene->aovDefaultValues });
      buffers.push_back({ .binding = rp::BINDING_INDEX_AOV_CLEAR_VALUES_I, .buffer = scene->aovDefaultValues });

      std::array<uint32_t, size_t(GiAovId::COUNT)> aovBindingIndices = {
        rp::BINDING_INDEX_AOV_COLOR,
        rp::BINDING_INDEX_AOV_NORMAL,
        rp::BINDING_INDEX_AOV_NEE,
        rp::BINDING_INDEX_AOV_BARYCENTRICS,
        rp::BINDING_INDEX_AOV_TEXCOORDS,
        rp::BINDING_INDEX_AOV_BOUNCES,
        rp::BINDING_INDEX_AOV_CLOCK_CYCLES,
        rp::BINDING_INDEX_AOV_OPACITY,
        rp::BINDING_INDEX_AOV_TANGENTS,
        rp::BINDING_INDEX_AOV_BITANGENTS,
        rp::BINDING_INDEX_AOV_THIN_WALLED,
        rp::BINDING_INDEX_AOV_OBJECT_ID,
        rp::BINDING_INDEX_AOV_DEPTH,
        rp::BINDING_INDEX_AOV_FACE_ID,
        rp::BINDING_INDEX_AOV_INSTANCE_ID,
        rp::BINDING_INDEX_AOV_DOUBLE_SIDED,
        rp::BINDING_INDEX_AOV_ALBEDO
      };

      for (const GiAovBinding& binding : params.aovBindings)
      {
        uint32_t bindingIndex = aovBindingIndices[int(binding.aovId)];
        buffers.push_back({ .binding = bindingIndex, .buffer = binding.renderBuffer->deviceMem });
      }

      size_t imageCount = shaderCache->imageBindings.size() + 2/* dome lights */;

      std::vector<CgpuImageBinding> images;
      images.reserve(imageCount);

      CgpuSamplerBinding sampler = { .binding = rp::BINDING_INDEX_SAMPLER, .sampler = s_texSampler };

      images.push_back({ .binding = rp::BINDING_INDEX_TEXTURES, .image = scene->fallbackDomeLightTexture,
                         .index = scene->domeLightsAllocation.offset + 0 });
      images.push_back({ .binding = rp::BINDING_INDEX_TEXTURES, .image = *scene->domeLightTexture,
                         .index = scene->domeLightsAllocation.offset + 1 });

      for (const GiImageBinding& b : shaderCache->imageBindings)
      {
        images.push_back({ .binding = rp::BINDING_INDEX_TEXTURES, .image = *b.image, .index = b.index });
      }

      CgpuTlasBinding as = { .binding = rp::BINDING_INDEX_SCENE_AS, .as = bvh->tlas };

      CgpuBindings bindings0 = {
        .bufferCount = (uint32_t) buffers.size(),
        .buffers = buffers.data(),
        .samplerCount = imageCount ? 1u : 0u,
        .samplers = &sampler,
        .tlasCount = 1u,
        .tlases = &as
      };

      CgpuBindings bindings1 = { .imageCount = (uint32_t) images.size(), .images = images.data() };
      CgpuBindings bindings2 = { .imageCount = (uint32_t) images.size(), .images = images.data() };

      if (shaderCache->imageBindings.size() > rp::MAX_TEXTURE_COUNT)
      {
        GI_FATAL("max number of textures exceeded");
      }

      cgpuCmdTransitionShaderImageLayouts(s_ctx, commandBuffer, shaderCache->rgenShader, 1/*descriptorSetIndex*/, (uint32_t) images.size(), images.data());

      cgpuUpdateBindSet(s_ctx, shaderCache->bindSets[0], &bindings0);
      cgpuUpdateBindSet(s_ctx, shaderCache->bindSets[1], &bindings1);
      cgpuUpdateBindSet(s_ctx, shaderCache->bindSets[2], &bindings2);

      scene->dirtyFlags &= ~GiSceneDirtyFlags::DirtyBindSets;
    }

    // Update uniforms
    uint32_t uniformOffset;
    {
      auto camForward = glm::normalize(glm::make_vec3(params.camera.forward));
      auto camUp = glm::normalize(glm::make_vec3(params.camera.up));

      float lensRadius = 0.0f;
      if (params.camera.fStop > 0.0f)
      {
        lensRadius = params.camera.focalLength / (2.0f * params.camera.fStop);
      }

      glm::quat domeLightRotation = scene->domeLight ? scene->domeLight->rotation : glm::quat()/* doesn't matter, uniform color */;
      glm::vec3 domeLightEmissionMultiplier = scene->domeLight ? scene->domeLight->baseEmission : glm::vec3(1.0f);
      uint32_t domeLightDiffuseSpecularPacked = glm::packHalf2x16(scene->domeLight ? glm::vec2(scene->domeLight->diffuse, scene->domeLight->specular) : glm::vec2(1.0f));

      uint32_t totalLightCount = scene->sphereLights.elementCount() + scene->distantLights.elementCount() +
        scene->rectLights.elementCount() + scene->diskLights.elementCount();

      auto uniformData = s_bumpAlloc->alloc<rp::UniformData>();

      *uniformData.cpuPtr = {
        .domeLightRotation = glm::make_vec4(&domeLightRotation[0]),
        .domeLightEmissionMultiplier = domeLightEmissionMultiplier,
        .domeLightDiffuseSpecularPacked = domeLightDiffuseSpecularPacked,
        .maxTextureIndex = shaderCache->maxTextureIndex,
        .sphereLightCount = scene->sphereLights.elementCount(),
        .distantLightCount = scene->distantLights.elementCount(),
        .rectLightCount = scene->rectLights.elementCount(),
        .diskLightCount = scene->diskLights.elementCount(),
        .totalLightCount = totalLightCount,
        .metersPerSceneUnit = renderSettings.metersPerSceneUnit,
        .maxVolumeWalkLength = renderSettings.maxVolumeWalkLength,
        .cameraPosition = glm::make_vec3(params.camera.position),
        .imageDims = ((imageHeight << 16) | imageWidth),
        .cameraForward = camForward,
        .focusDistance = params.camera.focusDistance,
        .cameraUp = camUp,
        .cameraVFoV = params.camera.vfov,
        .sampleOffset = scene->sampleOffset,
        .lensRadius = lensRadius,
        .spp = renderSettings.spp,
        .invSampleCount = 1.0f / float(scene->sampleOffset + renderSettings.spp),
        .maxSampleValue = renderSettings.maxSampleValue,
        .maxBouncesAndRrBounceOffset = ((renderSettings.maxBounces << 16) | renderSettings.rrBounceOffset),
        .rrInvMinTermProb = renderSettings.rrInvMinTermProb,
        .lightIntensityMultiplier = renderSettings.lightIntensityMultiplier,
        .clipRangePacked = glm::packHalf2x16(glm::vec2(params.camera.clipStart, params.camera.clipEnd)),
        .sensorExposure = params.camera.exposure,
      };

      uniformOffset = uniformData.bufferOffset;
    }

    // Bind pipeline and descriptor sets
    {
      std::array<uint32_t, 1> dynamicOffsets { uniformOffset };
      cgpuCmdBindPipeline(s_ctx, commandBuffer, shaderCache->pipeline,
                          shaderCache->bindSets.data(), uint32_t(shaderCache->bindSets.size()),
                          uint32_t(dynamicOffsets.size()), dynamicOffsets.data());
    }

    // Trace rays
    cgpuCmdTraceRays(s_ctx, commandBuffer, imageWidth, imageHeight);

    // Copy device to host memory
    {
      GbSmallVector<CgpuBufferMemoryBarrier, 5> preBarriers;
      GbSmallVector<CgpuBufferMemoryBarrier, 5> postBarriers;

      preBarriers.resize(params.aovBindings.size());
      postBarriers.resize(params.aovBindings.size());

      for (size_t i = 0; i < params.aovBindings.size(); i++)
      {
        const GiAovBinding& binding = params.aovBindings[i];
        GiRenderBuffer* renderBuffer = binding.renderBuffer;

        preBarriers[i] = CgpuBufferMemoryBarrier {
          .buffer = renderBuffer->deviceMem,
          .srcStageMask = CgpuPipelineStage::RayTracingShader,
          .srcAccessMask = CgpuMemoryAccess::ShaderWrite,
          .dstStageMask = CgpuPipelineStage::Transfer,
          .dstAccessMask = CgpuMemoryAccess::TransferRead
        };

        postBarriers[i] = CgpuBufferMemoryBarrier {
          .buffer = renderBuffer->hostMem,
          .srcStageMask = CgpuPipelineStage::Transfer,
          .srcAccessMask = CgpuMemoryAccess::TransferWrite,
          .dstStageMask = CgpuPipelineStage::Host,
          .dstAccessMask = CgpuMemoryAccess::HostRead
        };
      }

      CgpuPipelineBarrier preBarrier = {
        .bufferBarrierCount = (uint32_t) preBarriers.size(),
        .bufferBarriers = preBarriers.data()
      };

      cgpuCmdPipelineBarrier(s_ctx, commandBuffer, &preBarrier);

      for (const GiAovBinding& binding : params.aovBindings)
      {
        GiRenderBuffer* renderBuffer = binding.renderBuffer;

        cgpuCmdCopyBuffer(s_ctx, commandBuffer, renderBuffer->deviceMem, 0, renderBuffer->hostMem);
      }

      CgpuPipelineBarrier postBarrier = {
        .bufferBarrierCount = (uint32_t) postBarriers.size(),
        .bufferBarriers = postBarriers.data()
      };

      cgpuCmdPipelineBarrier(s_ctx, commandBuffer, &postBarrier);
    }

    // Submit command buffer
    cgpuEndCommandBuffer(s_ctx, commandBuffer);

    if (!cgpuCreateSemaphore(s_ctx, &semaphore))
      goto cleanup;

    signalSemaphoreInfo = { .semaphore = semaphore, .value = 1 };
    cgpuSubmitCommandBuffer(s_ctx, commandBuffer, 1, &signalSemaphoreInfo);

    waitSemaphoreInfo = { .semaphore = semaphore, .value = 1 };
    if (!cgpuWaitSemaphores(s_ctx, 1, &waitSemaphoreInfo))
      goto cleanup;

    s_deleteQueue->nextFrame();
    s_deleteQueue->housekeep();

    for (const GiAovBinding& binding : params.aovBindings)
    {
      if (binding.aovId == GiAovId::ClockCycles)
      {
        _EncodeRenderBufferAsHeatmap(binding.renderBuffer);
      }
    }

    scene->sampleOffset += renderSettings.spp;

    result = GiStatus::Ok;

cleanup:
    cgpuDestroySemaphore(s_ctx, semaphore);
    cgpuDestroyCommandBuffer(s_ctx, commandBuffer);

    return result;
  }

  GiScene* giCreateScene()
  {
    CgpuImage fallbackDomeLightTexture;
    if (!cgpuCreateImage(s_ctx, { .width = 1, .height = 1 }, &fallbackDomeLightTexture))
    {
      return nullptr;
    }

    GiScene* scene = new GiScene{
      .sphereLights = GgpuDenseDataStore(s_ctx, *s_stager, *s_deleteQueue, sizeof(rp::SphereLight), 64),
      .distantLights = GgpuDenseDataStore(s_ctx, *s_stager, *s_deleteQueue, sizeof(rp::DistantLight), 64),
      .rectLights = GgpuDenseDataStore(s_ctx, *s_stager, *s_deleteQueue, sizeof(rp::RectLight), 64),
      .diskLights = GgpuDenseDataStore(s_ctx, *s_stager, *s_deleteQueue, sizeof(rp::DiskLight), 64),
      .fallbackDomeLightTexture = fallbackDomeLightTexture,
    };

    scene->domeLightsAllocation = scene->texAllocator.allocate(2); // primary + fallback
    if (scene->domeLightsAllocation.offset == OffsetAllocator::Allocation::NO_SPACE)
    {
      GI_FATAL("max number of textures exceeded");
    }

    return scene;
  }

  void giDestroyScene(GiScene* scene)
  {
    if (scene->bvh)
    {
      _giDestroyBvh(scene->bvh);
    }
    if (scene->shaderCache)
    {
      _giDestroyShaderCache(scene->shaderCache);
    }
    if (scene->domeLight)
    {
      scene->domeLightTexture.reset();
    }
    if (scene->aovDefaultValues.handle)
    {
      cgpuDestroyBuffer(s_ctx, scene->aovDefaultValues);
    }
    cgpuDestroyImage(s_ctx, scene->fallbackDomeLightTexture);
    delete scene;
  }

  GiSphereLight* giCreateSphereLight(GiScene* scene)
  {
    std::lock_guard guard(scene->mutex);

    auto* light = new GiSphereLight;
    light->scene = scene;
    light->gpuHandle = scene->sphereLights.allocate();

    auto* data = scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->pos[0] = 0.0f;
    data->pos[1] = 0.0f;
    data->pos[2] = 0.0f;
    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(1.0f));
    data->baseEmission[0] = 0.0f;
    data->baseEmission[1] = 0.0f;
    data->baseEmission[2] = 0.0f;
    data->area = 1.0f;
    data->radiusXYZ[0] = 0.5f;
    data->radiusXYZ[1] = 0.5f;
    data->radiusXYZ[2] = 0.5f;

    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    return light;
  }

  void giDestroySphereLight(GiScene* scene, GiSphereLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->sphereLights.free(light->gpuHandle);
    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    delete light;
  }

  void giSetSphereLightPosition(GiSphereLight* light, float* pos)
  {
    auto* data = light->scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->pos[0] = pos[0];
    data->pos[1] = pos[1];
    data->pos[2] = pos[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetSphereLightBaseEmission(GiSphereLight* light, float* rgb)
  {
    auto* data = light->scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetSphereLightRadius(GiSphereLight* light, float radiusX, float radiusY, float radiusZ)
  {
    float ab = powf(radiusX * radiusY, 1.6f);
    float ac = powf(radiusX * radiusZ, 1.6f);
    float bc = powf(radiusY * radiusZ, 1.6f);
    float area = float(powf((ab + ac + bc) / 3.0f, 1.0f / 1.6f) * 4.0f * M_PI);

    auto* data = light->scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->radiusXYZ[0] = radiusX;
    data->radiusXYZ[1] = radiusY;
    data->radiusXYZ[2] = radiusZ;
    data->area = area;

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetSphereLightDiffuseSpecular(GiSphereLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  GiDistantLight* giCreateDistantLight(GiScene* scene)
  {
    std::lock_guard guard(scene->mutex);

    auto* light = new GiDistantLight;
    light->scene = scene;
    light->gpuHandle = scene->distantLights.allocate();

    auto* data = scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->direction[0] = 0.0f;
    data->direction[1] = 0.0f;
    data->direction[2] = 0.0f;
    data->angle = 0.0f;
    data->baseEmission[0] = 0.0f;
    data->baseEmission[1] = 0.0f;
    data->baseEmission[2] = 0.0f;
    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(1.0f));
    data->invPdf = 1.0f;

    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    return light;
  }

  void giDestroyDistantLight(GiScene* scene, GiDistantLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->distantLights.free(light->gpuHandle);
    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    delete light;
  }

  void giSetDistantLightDirection(GiDistantLight* light, float* direction)
  {
    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->direction[0] = direction[0];
    data->direction[1] = direction[1];
    data->direction[2] = direction[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDistantLightBaseEmission(GiDistantLight* light, float* rgb)
  {
    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDistantLightAngle(GiDistantLight* light, float angle)
  {
    float halfAngle = 0.5f * angle;
    float invPdf = (halfAngle > 0.0f) ? float(2.0f * M_PI * (1.0f - cosf(halfAngle))) : 1.0f;

    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->angle = angle;
    data->invPdf = invPdf;

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDistantLightDiffuseSpecular(GiDistantLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  GiRectLight* giCreateRectLight(GiScene* scene)
  {
    std::lock_guard guard(scene->mutex);

    auto* light = new GiRectLight;
    light->scene = scene;
    light->gpuHandle = scene->rectLights.allocate();

    uint32_t t0packed = _EncodeDirection(glm::vec3(1.0f, 0.0f, 0.0f));
    uint32_t t1packed = _EncodeDirection(glm::vec3(0.0f, 1.0f, 0.0f));

    auto* data = scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->origin[0] = 0.0f;
    data->origin[1] = 0.0f;
    data->origin[2] = 0.0f;
    data->width = 1.0f;
    data->baseEmission[0] = 0.0f;
    data->baseEmission[1] = 0.0f;
    data->baseEmission[2] = 0.0f;
    data->height = 1.0f;
    data->tangentFramePacked = glm::uvec2(t0packed, t1packed);
    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(1.0f));

    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    return light;
  }

  void giDestroyRectLight(GiScene* scene, GiRectLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->rectLights.free(light->gpuHandle);
    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    delete light;
  }

  void giSetRectLightOrigin(GiRectLight* light, float* origin)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->origin[0] = origin[0];
    data->origin[1] = origin[1];
    data->origin[2] = origin[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetRectLightTangents(GiRectLight* light, float* t0, float* t1)
  {
    uint32_t t0packed = _EncodeDirection(glm::make_vec3(t0));
    uint32_t t1packed = _EncodeDirection(glm::make_vec3(t1));

    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->tangentFramePacked = glm::uvec2(t0packed, t1packed);

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetRectLightBaseEmission(GiRectLight* light, float* rgb)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetRectLightDimensions(GiRectLight* light, float width, float height)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->width = width;
    data->height = height;

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetRectLightDiffuseSpecular(GiRectLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  GiDiskLight* giCreateDiskLight(GiScene* scene)
  {
    std::lock_guard guard(scene->mutex);

    auto* light = new GiDiskLight;
    light->scene = scene;
    light->gpuHandle = scene->diskLights.allocate();

    uint32_t t0packed = _EncodeDirection(glm::vec3(1.0f, 0.0f, 0.0f));
    uint32_t t1packed = _EncodeDirection(glm::vec3(0.0f, 1.0f, 0.0f));

    auto* data = scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->origin[0] = 0.0f;
    data->origin[1] = 0.0f;
    data->origin[2] = 0.0f;
    data->radiusX = 0.5f;
    data->baseEmission[0] = 0.0f;
    data->baseEmission[1] = 0.0f;
    data->baseEmission[2] = 0.0f;
    data->radiusY = 0.5f;
    data->tangentFramePacked = glm::uvec2(t0packed, t1packed);
    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(1.0f));

    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    return light;
  }

  void giDestroyDiskLight(GiScene* scene, GiDiskLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->diskLights.free(light->gpuHandle);
    scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;

    delete light;
  }

  void giSetDiskLightOrigin(GiDiskLight* light, float* origin)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->origin[0] = origin[0];
    data->origin[1] = origin[1];
    data->origin[2] = origin[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDiskLightTangents(GiDiskLight* light, float* t0, float* t1)
  {
    uint32_t t0packed = _EncodeDirection(glm::make_vec3(t0));
    uint32_t t1packed = _EncodeDirection(glm::make_vec3(t1));

    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->tangentFramePacked = glm::uvec2(t0packed, t1packed);

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDiskLightBaseEmission(GiDiskLight* light, float* rgb)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDiskLightRadius(GiDiskLight* light, float radiusX, float radiusY)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->radiusX = radiusX;
    data->radiusY = radiusY;

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDiskLightDiffuseSpecular(GiDiskLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));

    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  GiDomeLight* giCreateDomeLight(GiScene* scene, const char* filePath)
  {
    std::lock_guard guard(scene->mutex);

    auto* light = new GiDomeLight;
    light->scene = scene;
    light->textureFilePath = filePath;
    return light;
  }

  void giDestroyDomeLight(GiDomeLight* light)
  {
    std::lock_guard guard(light->scene->mutex);
    delete light;
  }

  void giSetDomeLightRotation(GiDomeLight* light, float* quat)
  {
    light->rotation = glm::make_quat(quat);
    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDomeLightBaseEmission(GiDomeLight* light, float* rgb)
  {
    light->baseEmission = glm::make_vec3(rgb);
    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  void giSetDomeLightDiffuseSpecular(GiDomeLight* light, float diffuse, float specular)
  {
    light->diffuse = diffuse;
    light->specular = specular;
    light->scene->dirtyFlags |= GiSceneDirtyFlags::DirtyFramebuffer;
  }

  GiRenderBuffer* giCreateRenderBuffer(uint32_t width, uint32_t height, GiRenderBufferFormat format)
  {
    uint32_t stride = _GiRenderBufferFormatStride(format);
    uint32_t size = width * height * stride;

    return new GiRenderBuffer {
      .width = width,
      .height = height,
      .size = size
    };
  }

  void giDestroyRenderBuffer(GiRenderBuffer* renderBuffer)
  {
    if (renderBuffer->deviceMem.handle)
    {
      s_deleteQueue->pushBack(renderBuffer->deviceMem);
    }
    if (renderBuffer->hostMem.handle)
    {
      s_deleteQueue->pushBack(renderBuffer->hostMem);
    }
    delete renderBuffer;
  }

  void* giGetRenderBufferMem(GiRenderBuffer* renderBuffer)
  {
    return renderBuffer->mappedHostMem;
  }
}
