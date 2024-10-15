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
#include <gtl/ggpu/DelayedResourceDestroyer.h>
#include <gtl/ggpu/DenseDataStore.h>
#include <gtl/cgpu/Cgpu.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#ifdef GI_SHADER_HOTLOADING
#include <efsw/efsw.hpp>
#endif
#include <gtl/mc/Material.h>
#include <gtl/mc/Frontend.h>
#include <gtl/mc/Runtime.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/Enum.h>

#include <MaterialXCore/Document.h>

namespace mx = MaterialX;

constexpr static const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

namespace gtl
{
  namespace rp = shader_interface::rp_main;

  class McRuntime;

  struct GiGpuBufferView
  {
    uint64_t offset;
    uint64_t size;
  };

  struct GiMeshCpuData
  {
    std::vector<GiFace> faces;
    std::vector<GiVertex> vertices;
  };

  struct GiMeshGpuData
  {
    CgpuBlas blas;
    CgpuBuffer payloadBuffer;
    rp::BlasPayload payload;
  };

  struct GiBvh
  {
    CgpuBuffer          blasPayloadsBuffer;
    GiScene*            scene;
    CgpuTlas            tlas;
  };

  struct GiShaderCache
  {
    uint32_t                       aovId = UINT32_MAX;
    bool                           domeLightCameraVisible;
    std::vector<CgpuShader>        hitShaders;
    std::vector<CgpuImage>         images2d;
    std::vector<CgpuImage>         images3d;
    std::vector<const GiMaterial*> materials;
    std::vector<CgpuShader>        missShaders;
    CgpuPipeline                   pipeline;
    bool                           hasPipelineClosestHitShader = false;
    bool                           hasPipelineAnyHitShader = false;
    CgpuShader                     rgenShader;
    bool                           resetSampleOffset = true;
  };

  struct GiMaterial
  {
    McMaterial* mcMat;
  };

  struct GiMesh
  {
    glm::mat3x4 transform;
    bool flipFacing;
    int id;
    std::vector<glm::mat3x4> instanceTransforms;
    const GiMaterial* material;
    GiScene* scene;
    GiMeshCpuData cpuData;
    std::optional<GiMeshGpuData> gpuData;
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
    Clean = 0,
    DirtyTlas,
    DirtyRtPipeline,
    All = ~0u
  };
  GB_DECLARE_ENUM_BITOPS(GiSceneDirtyFlags)

  struct GiScene
  {
    GgpuDenseDataStore sphereLights;
    GgpuDenseDataStore distantLights;
    GgpuDenseDataStore rectLights;
    GgpuDenseDataStore diskLights;
    CgpuImage domeLightTexture;
    GiDomeLight* domeLight = nullptr; // weak ptr
    glm::vec4 backgroundColor = glm::vec4(-1.0f); // used to initialize fallback dome light
    CgpuImage fallbackDomeLightTexture;
    std::unordered_set<GiBvh*> bvhs;
    std::unordered_set<GiMesh*> meshes;
    std::mutex mutex;
    GiSceneDirtyFlags dirtyFlags = GiSceneDirtyFlags::All;
  };

  struct GiRenderBuffer
  {
    CgpuBuffer buffer;
    CgpuBuffer stagingBuffer;
    uint32_t bufferWidth = 0;
    uint32_t bufferHeight = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t size = 0;
    uint32_t sampleOffset = 0;
  };

  bool s_cgpuInitialized = false;
  CgpuDevice s_device;
  CgpuPhysicalDeviceFeatures s_deviceFeatures;
  CgpuPhysicalDeviceProperties s_deviceProperties;
  CgpuSampler s_texSampler;
  std::unique_ptr<GgpuStager> s_stager;
  std::unique_ptr<GgpuDelayedResourceDestroyer> s_delayedResourceDestroyer;
  std::unique_ptr<GiGlslShaderGen> s_shaderGen;
  std::unique_ptr<McRuntime> s_mcRuntime;
  std::unique_ptr<McFrontend> s_mcFrontend;
  std::unique_ptr<GiMmapAssetReader> s_mmapAssetReader;
  std::unique_ptr<GiAggregateAssetReader> s_aggregateAssetReader;
  std::unique_ptr<GiTextureManager> s_texSys;
  std::atomic_bool s_forceShaderCacheInvalid = false;
  std::atomic_bool s_forceGeomCacheInvalid = false; // TODO: remove
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

  bool _giResizeRenderBufferIfNeeded(GiRenderBuffer* renderBuffer, uint32_t pixelStride)
  {
    uint32_t width = renderBuffer->width;
    uint32_t height = renderBuffer->height;
    uint32_t bufferSize = width * height * pixelStride;

    bool reallocBuffers = (renderBuffer->bufferWidth != width) ||
                          (renderBuffer->bufferHeight != height);

    if (!reallocBuffers)
    {
      return true;
    }

    if (renderBuffer->buffer.handle)
    {
      cgpuDestroyBuffer(s_device, renderBuffer->buffer);
      renderBuffer->buffer.handle = 0;
    }
    if (renderBuffer->stagingBuffer.handle)
    {
      cgpuDestroyBuffer(s_device, renderBuffer->stagingBuffer);
      renderBuffer->stagingBuffer.handle = 0;
    }

    if (width == 0 || height == 0)
    {
      return true;
    }

    GB_LOG("recreating output buffer with size {}x{} ({:.2f} MiB)", width, height, bufferSize * BYTES_TO_MIB);

    if (!cgpuCreateBuffer(s_device, {
                            .usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
                            .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                            .size = bufferSize,
                            .debugName = "RenderBuffer"
                          }, &renderBuffer->buffer))
    {
      return false;
    }

    if (!cgpuCreateBuffer(s_device, {
                            .usage = CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
                            .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
                            .size = bufferSize,
                            .debugName = "RenderBufferStaging"
                          }, &renderBuffer->stagingBuffer))
    {
      cgpuDestroyBuffer(s_device, renderBuffer->buffer);
      return false;
    }

    renderBuffer->bufferWidth = width;
    renderBuffer->bufferHeight = height;
    renderBuffer->size = bufferSize;

    return true;
  }

  void _PrintInitInfo(const GiInitParams& params)
  {
    GB_LOG("gatling {}.{}.{} built against MaterialX {}.{}.{}", GI_VERSION_MAJOR, GI_VERSION_MINOR, GI_VERSION_PATCH,
                                                                MATERIALX_MAJOR_VERSION, MATERIALX_MINOR_VERSION, MATERIALX_BUILD_VERSION);
    GB_LOG("> shader path: \"{}\"", params.shaderPath);
    GB_LOG("> MDL runtime path: \"{}\"", params.mdlRuntimePath);
    GB_LOG("> MDL search paths: {}", params.mdlSearchPaths);
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

    if (!cgpuInitialize("gatling", GI_VERSION_MAJOR, GI_VERSION_MINOR, GI_VERSION_PATCH))
      goto fail;

    s_cgpuInitialized = true;

    if (!cgpuCreateDevice(&s_device))
      goto fail;

    if (!cgpuGetPhysicalDeviceFeatures(s_device, &s_deviceFeatures))
      goto fail;

    if (!cgpuGetPhysicalDeviceProperties(s_device, &s_deviceProperties))
      goto fail;

    if (!cgpuCreateSampler(s_device, {
                            .addressModeU = CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                            .addressModeV = CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                            .addressModeW = CGPU_SAMPLER_ADDRESS_MODE_REPEAT
                          }, &s_texSampler))
    {
      goto fail;
    }

    s_stager = std::make_unique<GgpuStager>(s_device);
    if (!s_stager->allocate())
    {
      goto fail;
    }

    s_delayedResourceDestroyer = std::make_unique<GgpuDelayedResourceDestroyer>(s_device);

    s_mcRuntime = std::unique_ptr<McRuntime>(McLoadRuntime(params.mdlRuntimePath));
    if (!s_mcRuntime)
    {
      goto fail;
    }

    s_mcFrontend = std::make_unique<McFrontend>(params.mdlSearchPaths, mtlxStdLib, *s_mcRuntime);

    s_shaderGen = std::make_unique<GiGlslShaderGen>();
    if (!s_shaderGen->init(shaderPath, *s_mcRuntime))
    {
      goto fail;
    }

    s_mmapAssetReader = std::make_unique<GiMmapAssetReader>();
    s_aggregateAssetReader = std::make_unique<GiAggregateAssetReader>();
    s_aggregateAssetReader->addAssetReader(s_mmapAssetReader.get());

    s_texSys = std::make_unique<GiTextureManager>(s_device, *s_aggregateAssetReader, *s_stager);

#ifdef GI_SHADER_HOTLOADING
    s_fileWatcher = std::make_unique<efsw::FileWatcher>();
    s_fileWatcher->addWatch(shaderPath.data(), &s_shaderFileListener, true);
    s_fileWatcher->watch();
#endif

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
      cgpuDestroySampler(s_device, s_texSampler);
      s_texSampler = {};
    }
    if (s_delayedResourceDestroyer)
    {
      s_delayedResourceDestroyer->destroyAll();
      s_delayedResourceDestroyer.reset();
    }
    if (s_device.handle)
    {
      cgpuDestroyDevice(s_device);
      s_device = {};
    }
    if (s_cgpuInitialized)
    {
      cgpuTerminate();
      s_cgpuInitialized = false;
    }
    s_mcFrontend.reset();
    s_mcRuntime.reset();
  }

  void giRegisterAssetReader(GiAssetReader* reader)
  {
    s_aggregateAssetReader->addAssetReader(reader);
  }

  GiMaterial* giCreateMaterialFromMtlxStr(const char* str)
  {
    McMaterial* mcMat = s_mcFrontend->createFromMtlxStr(str);
    if (!mcMat)
    {
      return nullptr;
    }

    GiMaterial* mat = new GiMaterial;
    mat->mcMat = mcMat;
    return mat;
  }

  GiMaterial* giCreateMaterialFromMtlxDoc(const std::shared_ptr<void/*MaterialX::Document*/> doc)
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

    GiMaterial* mat = new GiMaterial;
    mat->mcMat = mcMat;
    return mat;
  }

  GiMaterial* giCreateMaterialFromMdlFile(const char* filePath, const char* subIdentifier)
  {
    McMaterial* mcMat = s_mcFrontend->createFromMdlFile(filePath, subIdentifier);
    if (!mcMat)
    {
      return nullptr;
    }

    GiMaterial* mat = new GiMaterial;
    mat->mcMat = mcMat;
    return mat;
  }

  void giDestroyMaterial(GiMaterial* mat)
  {
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
    GiMeshCpuData cpuData = {
      .faces = std::vector<GiFace>(&desc.faces[0], &desc.faces[desc.faceCount]),
      .vertices = std::vector<GiVertex>(&desc.vertices[0], &desc.vertices[desc.vertexCount])
    };

    GiMesh* mesh = new GiMesh {
      .transform = glm::mat3x4(1.0f),
      .flipFacing = desc.isLeftHanded,
      .material = nullptr,
      .id = desc.id,
      .scene = scene,
      .cpuData = cpuData
    };

    {
      std::lock_guard guard(scene->mutex);
      scene->meshes.insert(mesh);
    }
    return mesh;
  }

  void giSetMeshTransform(GiMesh* mesh, float transform[3][4])
  {
    memcpy(glm::value_ptr(mesh->transform), transform, sizeof(float) * 12);

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyTlas;
    }
  }

  void giSetMeshInstanceTransforms(GiMesh* mesh, uint32_t count, const float (*transforms)[4][4])
  {
    mesh->instanceTransforms.resize(count);
    for (uint32_t i = 0; i < count; i++)
    {
      mesh->instanceTransforms[i] = glm::mat3x4(glm::make_mat4((const float*) transforms[i]));
    }

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= GiSceneDirtyFlags::DirtyTlas;
    }
  }

  void giSetMeshMaterial(GiMesh* mesh, const GiMaterial* mat)
  {
    McMaterial* newMcMat = mat->mcMat;
    McMaterial* oldMcMat = mesh->material ? mesh->material->mcMat : nullptr;

    GiSceneDirtyFlags dirtyFlags = GiSceneDirtyFlags::DirtyRtPipeline;
    if (oldMcMat && newMcMat->hasCutoutTransparency != oldMcMat->hasCutoutTransparency)
    {
      // material data such as alpha is used in the BVH build; invalidate BVH
      dirtyFlags |= GiSceneDirtyFlags::DirtyTlas;
      mesh->gpuData.reset();
    }

    mesh->material = mat;

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->dirtyFlags |= dirtyFlags;
    }
  }

  void giDestroyMesh(GiMesh* mesh)
  {
    auto& gpuData = mesh->gpuData;

    if (gpuData.has_value())
    {
      cgpuDestroyBlas(s_device, gpuData->blas);
      cgpuDestroyBuffer(s_device, gpuData->payloadBuffer);
      gpuData.reset();
    }

    GiScene* scene = mesh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->meshes.erase(mesh);
    }
    delete mesh;
  }

  void _giBuildGeometryStructures(GiScene* scene,
                                  GiShaderCache* shaderCache,
                                  std::vector<CgpuBlasInstance>& blasInstances,
                                  std::vector<rp::BlasPayload>& blasPayloads,
                                  uint64_t& totalIndicesSize,
                                  uint64_t& totalVerticesSize)
  {
    for (auto it = scene->meshes.begin(); it != scene->meshes.end(); ++it)
    {
      GiMesh* mesh = *it;

      // Find material for SBT index (FIXME: find a better solution)
      uint32_t materialIndex = UINT32_MAX;
      for (uint32_t i = 0; i < shaderCache->materials.size(); i++)
      {
          if (shaderCache->materials[i] == mesh->material)
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
        const auto& data = mesh->cpuData;

        if (data.faces.empty())
        {
          continue;
        }

        // Payload buffer preamble
        rp::BlasPayloadBufferPreamble preamble = {
          .objectId = mesh->id
        };
        uint32_t preambleSize = sizeof(rp::BlasPayloadBufferPreamble);

        // Collect vertices
        std::vector<rp::FVertex> vertexData;
        std::vector<CgpuVertex> positionData;
        vertexData.resize(data.vertices.size());
        positionData.resize(data.vertices.size());

        for (uint32_t i = 0; i < positionData.size(); i++)
        {
          const GiVertex& cpuVert = data.vertices[i];
          uint32_t encodedNormal = _EncodeDirection(glm::make_vec3(cpuVert.norm));
          uint32_t encodedTangent = _EncodeDirection(glm::make_vec3(cpuVert.tangent));

          vertexData[i] = rp::FVertex{
            .field1 = { glm::make_vec3(cpuVert.pos), cpuVert.bitangentSign },
            .field2 = { *((float*)&encodedNormal), *((float*)&encodedTangent), cpuVert.u, cpuVert.v }
          };

          positionData[i] = CgpuVertex{ .x = cpuVert.pos[0], .y = cpuVert.pos[1], .z = cpuVert.pos[2] };
        }

        // Collect indices
        std::vector<uint32_t> indexData;
        indexData.reserve(data.faces.size() * 3);

        for (uint32_t i = 0; i < data.faces.size(); i++)
        {
          const auto* face = &data.faces[i];
          indexData.push_back(face->v_i[0]);
          indexData.push_back(face->v_i[1]);
          indexData.push_back(face->v_i[2]);
        }

        // Upload GPU data
        CgpuBlas blas;
        CgpuBuffer tmpPositionBuffer;
        CgpuBuffer tmpIndexBuffer;
        CgpuBuffer payloadBuffer;
        rp::BlasPayload payload;

        uint64_t indicesSize = indexData.size() * sizeof(uint32_t);
        uint64_t verticesSize = vertexData.size() * sizeof(rp::FVertex);

        uint64_t payloadBufferSize = preambleSize;
        uint64_t indexBufferOffset = giAlignBuffer(sizeof(rp::FVertex), indicesSize, &payloadBufferSize);
        uint64_t vertexBufferOffset = giAlignBuffer(sizeof(rp::FVertex), verticesSize, &payloadBufferSize);

        uint64_t tmpIndexBufferSize = indicesSize;
        uint64_t tmpPositionBufferSize = positionData.size() * sizeof(CgpuVertex);

        // Create data buffers
        if (!cgpuCreateBuffer(s_device, {
                                .usage = CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
                                .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                                .size = payloadBufferSize,
                                .debugName = "BlasPayloadBuffer"
                              }, &payloadBuffer))
        {
          GB_ERROR("failed to allocate BLAS payload buffer memory");
          goto fail_cleanup;
        }

        if (!cgpuCreateBuffer(s_device, {
                                .usage = CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
                                .size = tmpPositionBufferSize,
                                .debugName = "BlasVertexPositionsTmp"
                              }, &tmpPositionBuffer))
        {
          GB_ERROR("failed to allocate BLAS temp vertex position memory");
          goto fail_cleanup;
        }

        if (!cgpuCreateBuffer(s_device, {
                                .usage = CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
                                .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
                                .size = tmpIndexBufferSize,
                                .debugName = "BlasIndicesTmp"
                              }, &tmpIndexBuffer))
        {
          GB_ERROR("failed to allocate BLAS temp indices memory");
          goto fail_cleanup;
        }

        // Copy data to GPU
        {
          void* mappedMem;

          cgpuMapBuffer(s_device, tmpPositionBuffer, &mappedMem);
          memcpy(mappedMem, positionData.data(), tmpPositionBufferSize);
          cgpuUnmapBuffer(s_device, tmpPositionBuffer);

          cgpuMapBuffer(s_device, tmpIndexBuffer, &mappedMem);
          memcpy(mappedMem, indexData.data(), tmpIndexBufferSize);
          cgpuUnmapBuffer(s_device, tmpIndexBuffer);
        }

        if (!s_stager->stageToBuffer((uint8_t*) &preamble, preambleSize, payloadBuffer, 0) ||
            !s_stager->stageToBuffer((uint8_t*) indexData.data(), indicesSize, payloadBuffer, indexBufferOffset) ||
            !s_stager->stageToBuffer((uint8_t*) vertexData.data(), verticesSize, payloadBuffer, vertexBufferOffset))
        {
          GB_ERROR("failed to stage BLAS data");
          goto fail_cleanup;
        }

        s_stager->flush();

        // Build BLAS
        {
          const GiMaterial* material = shaderCache->materials[materialIndex];

          bool blasCreated = cgpuCreateBlas(s_device, {
                                              .vertexBuffer = tmpPositionBuffer,
                                              .indexBuffer = tmpIndexBuffer,
                                              .maxVertex = (uint32_t) positionData.size(),
                                              .triangleCount = (uint32_t) indexData.size() / 3,
                                              .isOpaque = !material->mcMat->hasCutoutTransparency
                                            }, &blas);

          if (!blasCreated)
          {
            GB_ERROR("failed to allocate BLAS vertex memory");
            goto fail_cleanup;
          }
        }

        cgpuDestroyBuffer(s_device, tmpPositionBuffer);
        tmpPositionBuffer.handle = 0;
        cgpuDestroyBuffer(s_device, tmpIndexBuffer);
        tmpIndexBuffer.handle = 0;

        // Append BLAS payload data
        {
          uint64_t payloadBufferAddress = cgpuGetBufferAddress(s_device, payloadBuffer);
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

          uint64_t vertexBufferSize = (vertexBufferOffset/* account for align */ - indexBufferOffset/* account for preamble */);
          payload = rp::BlasPayload{
            .bufferAddress = payloadBufferAddress,
            .vertexOffset = uint32_t(vertexBufferSize / sizeof(rp::FVertex)), // offset to skip index buffer
            .bitfield = bitfield
          };
        }

        // Append BLAS for lifetime management
        {
          mesh->gpuData = GiMeshGpuData{
            .blas = blas,
            .payloadBuffer = payloadBuffer,
            .payload = payload
          };
        }

        // (we ignore padding and the preamble in the reporting, but they are negligible)
        totalVerticesSize += verticesSize;
        totalIndicesSize += indicesSize;

        if (false) // not executed in success case
        {
fail_cleanup:
          if (payloadBuffer.handle)
            cgpuDestroyBuffer(s_device, payloadBuffer);
          if (tmpPositionBuffer.handle)
            cgpuDestroyBuffer(s_device, tmpPositionBuffer);
          if (tmpIndexBuffer.handle)
            cgpuDestroyBuffer(s_device, tmpIndexBuffer);
          if (blas.handle)
            cgpuDestroyBlas(s_device, blas);

          continue;
        }
      }

      const auto& data = mesh->gpuData;
      if (!data.has_value())
      {
        continue; // invalid geometry or an error occurred
      }

      for (const glm::mat3x4& t : mesh->instanceTransforms)
      {
        // Create BLAS instance for TLAS.
        glm::mat3x4 transform = glm::mat3x4(glm::mat4(mesh->transform) * glm::mat4(t));

        CgpuBlasInstance blasInstance;
        blasInstance.as = data->blas;
        blasInstance.hitGroupIndex = materialIndex * 2; // always two hit groups per material: regular & shadow
        blasInstance.instanceCustomIndex = uint32_t(blasPayloads.size());
        memcpy(blasInstance.transform, glm::value_ptr(transform), sizeof(float) * 12);

        blasInstances.push_back(blasInstance);
        blasPayloads.push_back(data->payload);
      }
    }
  }

  GiBvh* giCreateBvh(GiScene* scene, const GiBvhParams& params)
  {
    s_forceGeomCacheInvalid = false; // TODO: remove

    GiBvh* bvh = nullptr;

    GB_LOG("creating bvh..");
    fflush(stdout);

    // Build BLASes.
    CgpuTlas tlas;
    std::vector<CgpuBlasInstance> blasInstances;
    std::vector<rp::BlasPayload> blasPayloads;
    uint64_t indicesSize = 0;
    uint64_t verticesSize = 0;
    CgpuBuffer blasPayloadsBuffer;

    _giBuildGeometryStructures(scene, params.shaderCache, blasInstances, blasPayloads, indicesSize, verticesSize);

    GB_LOG("BLAS build finished");
    GB_LOG("> {} unique BLAS", blasPayloads.size());
    GB_LOG("> {} BLAS instances", blasInstances.size());
    GB_LOG("> {:.2f} MiB total indices", indicesSize * BYTES_TO_MIB);
    GB_LOG("> {:.2f} MiB total vertices", verticesSize * BYTES_TO_MIB);

    // Create TLAS.
    {
      if (!cgpuCreateTlas(s_device, {
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

      if (!cgpuCreateBuffer(s_device, {
                              .usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
                              .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
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

    // Fill cache struct.
    bvh = new GiBvh;
    bvh->blasPayloadsBuffer = blasPayloadsBuffer;
    bvh->scene = scene;
    bvh->tlas = tlas;

    {
      std::lock_guard guard(scene->mutex);
      scene->bvhs.insert(bvh);
    }

cleanup:
    if (!bvh)
    {
      assert(false);
      if (blasPayloadsBuffer.handle)
      {
        cgpuDestroyBuffer(s_device, blasPayloadsBuffer);
      }
      if (tlas.handle)
      {
        cgpuDestroyTlas(s_device, tlas);
      }
    }
    return bvh;
  }

  void giDestroyBvh(GiBvh* bvh)
  {
    cgpuDestroyTlas(s_device, bvh->tlas);
    cgpuDestroyBuffer(s_device, bvh->blasPayloadsBuffer);

    GiScene* scene = bvh->scene;
    {
      std::lock_guard guard(scene->mutex);
      scene->bvhs.erase(bvh);
    }
    delete bvh;
  }

  // FIXME: move this into the GiScene struct - also, want to rebuild with cached data at shader granularity
  bool giShaderCacheNeedsRebuild()
  {
    return s_forceShaderCacheInvalid;
  }
  // TODO: remove
  bool giGeomCacheNeedsRebuild()
  {
    return s_forceGeomCacheInvalid;
  }

  GiShaderCache* giCreateShaderCache(const GiShaderCacheParams& params)
  {
    s_forceShaderCacheInvalid = false;

    bool clockCyclesAov = params.aovId == GiAovId::ClockCycles;

    if (clockCyclesAov && !s_deviceFeatures.shaderClock)
    {
      GB_ERROR("unsupported AOV - device feature missing");
      return nullptr;
    }

    GiScene* scene = params.scene;

    std::vector<const GiMaterial*> materials;
    materials.reserve(scene->meshes.size());
    for (auto* m : scene->meshes)
    {
      materials.push_back(m->material);
    }

    GB_LOG("material count: {}", materials.size());
    GB_LOG("creating shader cache..");
    fflush(stdout);

    GiShaderCache* cache = nullptr;
    CgpuPipeline pipeline;
    CgpuShader rgenShader;
    std::vector<CgpuShader> missShaders;
    std::vector<CgpuShader> hitShaders;
    std::vector<CgpuImage> images2d;
    std::vector<CgpuImage> images3d;
    std::vector<CgpuRtHitGroup> hitGroups;
    std::vector<McTextureDescription> textureDescriptions;
    bool hasPipelineClosestHitShader = false;
    bool hasPipelineAnyHitShader = false;

    uint32_t diskLightCount = scene->diskLights.elementCount();
    uint32_t distantLightCount = scene->distantLights.elementCount();
    uint32_t rectLightCount = scene->rectLights.elementCount();
    uint32_t sphereLightCount = scene->sphereLights.elementCount();
    uint32_t totalLightCount = diskLightCount + distantLightCount + rectLightCount + sphereLightCount;

    bool nextEventEstimation = (params.nextEventEstimation && totalLightCount > 0);

    GiGlslShaderGen::CommonShaderParams commonParams = {
      .aovId = (int) params.aovId,
      .diskLightCount = diskLightCount,
      .distantLightCount = distantLightCount,
      .mediumStackSize = params.mediumStackSize,
      .rectLightCount = rectLightCount,
      .sphereLightCount = sphereLightCount,
      .texCount2d = 2, // +1 fallback and +1 real dome light
      .texCount3d = 0
    };

    uint32_t& texCount2d = commonParams.texCount2d;
    uint32_t& texCount3d = commonParams.texCount3d;

    // Create per-material closest-hit shaders.
    //
    // This is done in multiple phases: first, GLSL is generated from MDL, and
    // texture information is extracted. The information is then used to generated
    // the descriptor sets for the pipeline. Lastly, the GLSL is stiched, #defines
    // are added, and the code is compiled to SPIR-V.
    {
      // 1. Generate GLSL from MDL
      struct HitShaderCompInfo
      {
        GiGlslShaderGen::MaterialGenInfo genInfo;
        uint32_t texOffset2d = 0;
        uint32_t texOffset3d = 0;
        std::vector<uint8_t> spv;
        std::vector<uint8_t> shadowSpv;
      };
      struct HitGroupCompInfo
      {
        HitShaderCompInfo closestHitInfo;
        std::optional<HitShaderCompInfo> anyHitInfo;
      };

      std::vector<HitGroupCompInfo> hitGroupCompInfos;
      hitGroupCompInfos.resize(materials.size());

      std::atomic_bool threadWorkFailed = false;
#pragma omp parallel for
      for (int i = 0; i < int(hitGroupCompInfos.size()); i++)
      {
        const McMaterial* material = materials[i]->mcMat;

        HitGroupCompInfo groupInfo;
        {
          GiGlslShaderGen::MaterialGenInfo genInfo;
          if (!s_shaderGen->generateMaterialShadingGenInfo(*material, genInfo))
          {
            threadWorkFailed = true;
            continue;
          }

          HitShaderCompInfo hitInfo;
          hitInfo.genInfo = genInfo;
          groupInfo.closestHitInfo = hitInfo;
        }
        if (material->hasCutoutTransparency)
        {
          GiGlslShaderGen::MaterialGenInfo genInfo;
          if (!s_shaderGen->generateMaterialOpacityGenInfo(*material, genInfo))
          {
            threadWorkFailed = true;
            continue;
          }

          HitShaderCompInfo hitInfo;
          hitInfo.genInfo = genInfo;
          groupInfo.anyHitInfo = hitInfo;
        }

        hitGroupCompInfos[i] = groupInfo;
      }
      if (threadWorkFailed)
      {
        goto cleanup;
      }

      // 2. Sum up texture resources & calculate per-material index offsets.
      for (HitGroupCompInfo& groupInfo : hitGroupCompInfos)
      {
        HitShaderCompInfo& closestHitShaderCompInfo = groupInfo.closestHitInfo;
        closestHitShaderCompInfo.texOffset2d = texCount2d;
        closestHitShaderCompInfo.texOffset3d = texCount3d;

        for (const McTextureDescription& tr : closestHitShaderCompInfo.genInfo.textureDescriptions)
        {
          (tr.is3dImage ? texCount3d : texCount2d)++;
          textureDescriptions.push_back(tr);
        }

        if (groupInfo.anyHitInfo)
        {
          HitShaderCompInfo& anyHitShaderCompInfo = *groupInfo.anyHitInfo;
          anyHitShaderCompInfo.texOffset2d = texCount2d;
          anyHitShaderCompInfo.texOffset3d = texCount3d;

          for (const McTextureDescription& tr : anyHitShaderCompInfo.genInfo.textureDescriptions)
          {
            (tr.is3dImage ? texCount3d : texCount2d)++;
            textureDescriptions.push_back(tr);
          }

          hasPipelineAnyHitShader |= true;
        }
      }

      hasPipelineClosestHitShader = hitGroupCompInfos.size() > 0;

      // 3. Generate final hit shader GLSL sources.
      threadWorkFailed = false;
#pragma omp parallel for
      for (int i = 0; i < int(hitGroupCompInfos.size()); i++)
      {
        const McMaterial* material = materials[i]->mcMat;

        HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

        // Closest hit
        {
          GiGlslShaderGen::ClosestHitShaderParams hitParams = {
            .baseFileName = "rp_main.chit",
            .commonParams = commonParams,
            .directionalBias = material->directionalBias,
            .enableSceneTransforms = material->requiresSceneTransforms,
            .cameraPositionSceneDataIndex = material->cameraPositionSceneDataIndex,
            .hasBackfaceBsdf = material->hasBackfaceBsdf,
            .hasBackfaceEdf = material->hasBackfaceEdf,
            .hasCutoutTransparency = material->hasCutoutTransparency,
            .hasVolumeAbsorptionCoeff = material->hasVolumeAbsorptionCoeff,
            .hasVolumeScatteringCoeff = material->hasVolumeScatteringCoeff,
            .isEmissive = material->isEmissive,
            .isThinWalled = material->isThinWalled,
            .nextEventEstimation = nextEventEstimation,
            .shadingGlsl = compInfo.closestHitInfo.genInfo.glslSource,
            .textureIndexOffset2d = compInfo.closestHitInfo.texOffset2d,
            .textureIndexOffset3d = compInfo.closestHitInfo.texOffset3d
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
            .enableSceneTransforms = material->requiresSceneTransforms,
            .cameraPositionSceneDataIndex = material->cameraPositionSceneDataIndex,
            .opacityEvalGlsl = compInfo.anyHitInfo->genInfo.glslSource,
            .textureIndexOffset2d = compInfo.anyHitInfo->texOffset2d,
            .textureIndexOffset3d = compInfo.anyHitInfo->texOffset3d
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

      // 4. Compile the shaders to SPIV-V. (FIXME: multithread - beware of shared cgpu resource stores)
      hitShaders.reserve(hitGroupCompInfos.size());
      hitGroups.reserve(hitGroupCompInfos.size() * 2);

      for (int i = 0; i < int(hitGroupCompInfos.size()); i++)
      {
        const HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

        // regular hit group
        {
          CgpuShader closestHitShader;
          {
            const std::vector<uint8_t>& spv = compInfo.closestHitInfo.spv;

            if (!cgpuCreateShader(s_device, {
                                    .size = spv.size(),
                                    .source = spv.data(),
                                    .stageFlags = CGPU_SHADER_STAGE_FLAG_CLOSEST_HIT
                                  }, &closestHitShader))
            {
              goto cleanup;
            }

            hitShaders.push_back(closestHitShader);
          }

          CgpuShader anyHitShader;
          if (compInfo.anyHitInfo)
          {
            const std::vector<uint8_t>& spv = compInfo.anyHitInfo->spv;

            if (!cgpuCreateShader(s_device, {
                                    .size = spv.size(),
                                    .source = spv.data(),
                                    .stageFlags = CGPU_SHADER_STAGE_FLAG_ANY_HIT
                                  }, &anyHitShader))
            {
              goto cleanup;
            }

            hitShaders.push_back(anyHitShader);
          }

          CgpuRtHitGroup hitGroup;
          hitGroup.closestHitShader = closestHitShader;
          hitGroup.anyHitShader = anyHitShader;
          hitGroups.push_back(hitGroup);
        }

        // shadow hit group
        {
          CgpuShader anyHitShader;

          if (compInfo.anyHitInfo)
          {
            const std::vector<uint8_t>& spv = compInfo.anyHitInfo->shadowSpv;

            if (!cgpuCreateShader(s_device, {
                                    .size = spv.size(),
                                    .source = spv.data(),
                                    .stageFlags = CGPU_SHADER_STAGE_FLAG_ANY_HIT
                                  }, &anyHitShader))
            {
              goto cleanup;
            }

            hitShaders.push_back(anyHitShader);
          }

          CgpuRtHitGroup hitGroup;
          hitGroup.anyHitShader = anyHitShader;
          hitGroups.push_back(hitGroup);
        }
      }
    }

    // Create ray generation shader.
    {
      GiGlslShaderGen::RaygenShaderParams rgenParams = {
        .commonParams = commonParams,
        .depthOfField = params.depthOfField,
        .filterImportanceSampling = params.filterImportanceSampling,
        .materialCount = uint32_t(materials.size()),
        .nextEventEstimation = nextEventEstimation,
        .progressiveAccumulation = params.progressiveAccumulation,
        .reorderInvocations = s_deviceFeatures.rayTracingInvocationReorder,
        .shaderClockExt = clockCyclesAov
      };

      std::vector<uint8_t> spv;
      if (!s_shaderGen->generateRgenSpirv("rp_main.rgen", rgenParams, spv))
      {
        goto cleanup;
      }

      if (!cgpuCreateShader(s_device, {
                              .size = spv.size(),
                              .source = spv.data(),
                              .stageFlags = CGPU_SHADER_STAGE_FLAG_RAYGEN
                            }, &rgenShader))
      {
        goto cleanup;
      }
    }

    // Create miss shaders.
    {
      GiGlslShaderGen::MissShaderParams missParams = {
        .commonParams = commonParams,
        .domeLightCameraVisible = params.domeLightCameraVisible
      };

      // regular miss shader
      {
        std::vector<uint8_t> spv;
        if (!s_shaderGen->generateMissSpirv("rp_main.miss", missParams, spv))
        {
          goto cleanup;
        }

        CgpuShader missShader;
        if (!cgpuCreateShader(s_device, {
                                .size = spv.size(),
                                .source = spv.data(),
                                .stageFlags = CGPU_SHADER_STAGE_FLAG_MISS
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
        if (!cgpuCreateShader(s_device, {
                                .size = spv.size(),
                                .source = spv.data(),
                                .stageFlags = CGPU_SHADER_STAGE_FLAG_MISS
                              }, &missShader))
        {
          goto cleanup;
        }

        missShaders.push_back(missShader);
      }
    }

    // Upload textures.
    if (textureDescriptions.size() > 0 && !s_texSys->loadTextureDescriptions(textureDescriptions, images2d, images3d))
    {
      goto cleanup;
    }
    assert(images2d.size() == (texCount2d - 2));
    assert(images3d.size() == texCount3d);

    // Create RT pipeline.
    {
      GB_LOG("creating RT pipeline..");
      fflush(stdout);

      if (!cgpuCreateRtPipeline(s_device, {
                                  .rgenShader = rgenShader,
                                  .missShaderCount = (uint32_t)missShaders.size(),
                                  .missShaders = missShaders.data(),
                                  .hitGroupCount = (uint32_t)hitGroups.size(),
                                  .hitGroups = hitGroups.data(),
                                }, &pipeline))
      {
        goto cleanup;
      }
    }

    cache = new GiShaderCache;
    cache->aovId = (int) params.aovId;
    cache->domeLightCameraVisible = params.domeLightCameraVisible;
    cache->hitShaders = std::move(hitShaders);
    cache->images2d = std::move(images2d);
    cache->images3d = std::move(images3d);
    cache->materials.resize(materials.size());
    for (uint32_t i = 0; i < cache->materials.size(); i++)
    {
      cache->materials[i] = materials[i];
    }
    cache->missShaders = missShaders;
    cache->pipeline = pipeline;
    cache->rgenShader = rgenShader;
    cache->hasPipelineClosestHitShader = hasPipelineClosestHitShader;
    cache->hasPipelineAnyHitShader = hasPipelineAnyHitShader;

cleanup:
    if (!cache)
    {
      s_texSys->destroyUncachedImages(images2d);
      s_texSys->destroyUncachedImages(images3d);
      if (rgenShader.handle)
      {
        cgpuDestroyShader(s_device, rgenShader);
      }
      for (CgpuShader shader : missShaders)
      {
        cgpuDestroyShader(s_device, shader);
      }
      for (CgpuShader shader : hitShaders)
      {
        cgpuDestroyShader(s_device, shader);
      }
      if (pipeline.handle)
      {
        cgpuDestroyPipeline(s_device, pipeline);
      }
    }
    return cache;
  }

  void giDestroyShaderCache(GiShaderCache* cache)
  {
    s_texSys->destroyUncachedImages(cache->images2d);
    s_texSys->destroyUncachedImages(cache->images3d);
    cgpuDestroyShader(s_device, cache->rgenShader);
    for (CgpuShader shader : cache->missShaders)
    {
      cgpuDestroyShader(s_device, shader);
    }
    for (CgpuShader shader : cache->hitShaders)
    {
      cgpuDestroyShader(s_device, shader);
    }
    cgpuDestroyPipeline(s_device, cache->pipeline);
    delete cache;
  }

  void giInvalidateFramebuffer()
  {
    s_resetSampleOffset = true;
  }

  void giInvalidateShaderCache()
  {
    s_forceShaderCacheInvalid = true;
  }

  void giInvalidateGeomCache()
  {
    s_forceGeomCacheInvalid = true;
  }

  GiStatus giRender(const GiRenderParams& params, float* rgbaImg)
  {
    s_stager->flush();

    const GiBvh* bvh = params.bvh;
    const GiShaderCache* shader_cache = params.shaderCache;
    GiScene* scene = params.scene;

    // Upload dome lights.
    glm::vec4 backgroundColor = glm::make_vec4(params.backgroundColor);
    if (backgroundColor != scene->backgroundColor)
    {
      glm::u8vec4 u8BgColor(backgroundColor * 255.0f);
      s_stager->stageToImage(glm::value_ptr(u8BgColor), 4, scene->fallbackDomeLightTexture, 1, 1);
      scene->backgroundColor = backgroundColor;
    }

    if (scene->domeLight != params.domeLight)
    {
      if (scene->domeLightTexture.handle &&
          scene->domeLightTexture.handle != scene->fallbackDomeLightTexture.handle)
      {
        s_texSys->evictAndDestroyCachedImage(scene->domeLightTexture);
        scene->domeLightTexture.handle = 0;
      }
      scene->domeLight = nullptr;

      GiDomeLight* domeLight = params.domeLight;
      if (domeLight)
      {
        const char* filePath = domeLight->textureFilePath.c_str();

        bool is3dImage = false;
        bool flushImmediately = false;
        if (!s_texSys->loadTextureFromFilePath(filePath, scene->domeLightTexture, is3dImage, flushImmediately))
        {
          GB_ERROR("unable to load dome light texture at {}", filePath);
        }
        else
        {
          scene->domeLight = domeLight;
        }
      }
    }
    if (!scene->domeLight)
    {
      // Use fallback texture in case no dome light is set. We still have an explicit binding
      // for the fallback texture because we need the background color in case the textured
      // dome light is not supposed to be seen by the camera ('domeLightCameraVisible' option).
      scene->domeLightTexture = scene->fallbackDomeLightTexture;
    }

    // Init state for goto error handling.
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

    // Set up output buffer.
    GiRenderBuffer* renderBuffer = params.renderBuffer;
    uint32_t imageWidth = renderBuffer->width;
    uint32_t imageHeight = renderBuffer->height;

    int compCount = 4;
    int pixelStride = compCount * sizeof(float);
    int pixelCount = imageWidth * imageHeight;

    if (!_giResizeRenderBufferIfNeeded(renderBuffer, pixelStride))
    {
      GB_ERROR("failed to resize render buffer!");
      return GiStatus::Error;
    }

    if (s_resetSampleOffset)
    {
      renderBuffer->sampleOffset = 0;
      s_resetSampleOffset = false;
    }

    // Set up GPU data.
    CgpuCommandBuffer commandBuffer;
    CgpuSemaphore semaphore;
    CgpuSignalSemaphoreInfo signalSemaphoreInfo;
    CgpuWaitSemaphoreInfo waitSemaphoreInfo;

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

    rp::PushConstants pushData = {
      .cameraPosition                 = glm::make_vec3(params.camera.position),
      .imageDims                      = ((imageHeight << 16) | imageWidth),
      .cameraForward                  = camForward,
      .focusDistance                  = params.camera.focusDistance,
      .cameraUp                       = camUp,
      .cameraVFoV                     = params.camera.vfov,
      .sampleOffset                   = renderBuffer->sampleOffset,
      .lensRadius                     = lensRadius,
      .sampleCount                    = params.spp,
      .maxSampleValue                 = params.maxSampleValue,
      .domeLightRotation              = glm::make_vec4(&domeLightRotation[0]),
      .domeLightEmissionMultiplier    = domeLightEmissionMultiplier,
      .domeLightDiffuseSpecularPacked = domeLightDiffuseSpecularPacked,
      .maxBouncesAndRrBounceOffset    = ((params.maxBounces << 16) | params.rrBounceOffset),
      .rrInvMinTermProb               = params.rrInvMinTermProb,
      .lightIntensityMultiplier       = params.lightIntensityMultiplier,
      .clipRangePacked                = glm::packHalf2x16(glm::vec2(params.camera.clipStart, params.camera.clipEnd)),
      .sensorExposure                 = params.camera.exposure,
      .maxVolumeWalkLength            = params.maxVolumeWalkLength
    };

    std::vector<CgpuBufferBinding> buffers;
    buffers.reserve(16);

    buffers.push_back({ .binding = rp::BINDING_INDEX_OUT_PIXELS, .buffer = renderBuffer->buffer });
    buffers.push_back({ .binding = rp::BINDING_INDEX_SPHERE_LIGHTS, .buffer = scene->sphereLights.buffer() });
    buffers.push_back({ .binding = rp::BINDING_INDEX_DISTANT_LIGHTS, .buffer = scene->distantLights.buffer() });
    buffers.push_back({ .binding = rp::BINDING_INDEX_RECT_LIGHTS, .buffer = scene->rectLights.buffer() });
    buffers.push_back({ .binding = rp::BINDING_INDEX_DISK_LIGHTS, .buffer = scene->diskLights.buffer() });
    buffers.push_back({ .binding = rp::BINDING_INDEX_BLAS_PAYLOADS, .buffer = bvh->blasPayloadsBuffer });

    size_t imageCount = shader_cache->images2d.size() + shader_cache->images3d.size() + 2/* dome lights */;

    std::vector<CgpuImageBinding> images;
    images.reserve(imageCount);

    CgpuSamplerBinding sampler = { .binding = rp::BINDING_INDEX_SAMPLER, .sampler = s_texSampler };

    images.push_back({ .binding = rp::BINDING_INDEX_TEXTURES_2D, .image = scene->fallbackDomeLightTexture, .index = 0 });
    images.push_back({ .binding = rp::BINDING_INDEX_TEXTURES_2D, .image = scene->domeLightTexture,         .index = 1 });

    for (uint32_t i = 0; i < shader_cache->images2d.size(); i++)
    {
      images.push_back({ .binding = rp::BINDING_INDEX_TEXTURES_2D,
                         .image = shader_cache->images2d[i],
                         .index = 2/* dome lights */ + i });
    }
    for (uint32_t i = 0; i < shader_cache->images3d.size(); i++)
    {
      images.push_back({ .binding = rp::BINDING_INDEX_TEXTURES_3D, .image = shader_cache->images3d[i], .index = i });
    }

    CgpuTlasBinding as = { .binding = rp::BINDING_INDEX_SCENE_AS, .as = bvh->tlas };

    CgpuBindings bindings = {
      .bufferCount = (uint32_t) buffers.size(),
      .buffers = buffers.data(),
      .imageCount = (uint32_t) images.size(),
      .images = images.data(),
      .samplerCount = imageCount ? 1u : 0u,
      .samplers = &sampler,
      .tlasCount = 1u,
      .tlases = &as
    };

    // Set up command buffer.
    if (!cgpuCreateCommandBuffer(s_device, &commandBuffer))
      goto cleanup;

    if (!cgpuBeginCommandBuffer(commandBuffer))
      goto cleanup;

    if (!cgpuCmdTransitionShaderImageLayouts(commandBuffer, shader_cache->rgenShader, (uint32_t) images.size(), images.data()))
      goto cleanup;

    if (!cgpuCmdUpdateBindings(commandBuffer, shader_cache->pipeline, &bindings))
      goto cleanup;

    if (!cgpuCmdBindPipeline(commandBuffer, shader_cache->pipeline))
      goto cleanup;

    // Trace rays.
    {
      CgpuShaderStageFlags pushShaderStages = CGPU_SHADER_STAGE_FLAG_RAYGEN | CGPU_SHADER_STAGE_FLAG_MISS;
      pushShaderStages |= shader_cache->hasPipelineClosestHitShader ? CGPU_SHADER_STAGE_FLAG_CLOSEST_HIT : 0;
      pushShaderStages |= shader_cache->hasPipelineAnyHitShader ? CGPU_SHADER_STAGE_FLAG_ANY_HIT : 0;

      if (!cgpuCmdPushConstants(commandBuffer, shader_cache->pipeline, pushShaderStages, sizeof(pushData), &pushData))
        goto cleanup;
    }

    if (!cgpuCmdTraceRays(commandBuffer, shader_cache->pipeline, imageWidth, imageHeight))
      goto cleanup;

    // Copy output buffer to staging buffer.
    {
      CgpuBufferMemoryBarrier bufferBarrier = {
        .buffer = renderBuffer->buffer,
        .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_RAY_TRACING_SHADER,
        .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE,
        .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_TRANSFER,
        .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ
      };
      CgpuPipelineBarrier barrier = {
        .bufferBarrierCount = 1,
        .bufferBarriers = &bufferBarrier
      };

      if (!cgpuCmdPipelineBarrier(commandBuffer, &barrier))
        goto cleanup;
    }

    if (!cgpuCmdCopyBuffer(commandBuffer, renderBuffer->buffer, 0, renderBuffer->stagingBuffer))
      goto cleanup;

    {
      CgpuBufferMemoryBarrier bufferBarrier = {
        .buffer = renderBuffer->stagingBuffer,
        .srcStageMask = CGPU_PIPELINE_STAGE_FLAG_TRANSFER,
        .srcAccessMask = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE,
        .dstStageMask = CGPU_PIPELINE_STAGE_FLAG_HOST,
        .dstAccessMask = CGPU_MEMORY_ACCESS_FLAG_HOST_READ
      };
      CgpuPipelineBarrier barrier = {
        .bufferBarrierCount = 1,
        .bufferBarriers = &bufferBarrier
      };

      if (!cgpuCmdPipelineBarrier(commandBuffer, &barrier))
        goto cleanup;
    }

    // Submit command buffer.
    if (!cgpuEndCommandBuffer(commandBuffer))
      goto cleanup;

    if (!cgpuCreateSemaphore(s_device, &semaphore))
      goto cleanup;

    signalSemaphoreInfo = { .semaphore = semaphore, .value = 1 };
    if (!cgpuSubmitCommandBuffer(s_device, commandBuffer, 1, &signalSemaphoreInfo))
      goto cleanup;

    waitSemaphoreInfo = { .semaphore = semaphore, .value = 1 };
    if (!cgpuWaitSemaphores(s_device, 1, &waitSemaphoreInfo))
      goto cleanup;

    s_delayedResourceDestroyer->nextFrame();

    // Read data from GPU to image.
    uint8_t* mapped_staging_mem;
    if (!cgpuMapBuffer(s_device, renderBuffer->stagingBuffer, (void**) &mapped_staging_mem))
      goto cleanup;

    memcpy(rgbaImg, mapped_staging_mem, renderBuffer->size);

    if (!cgpuUnmapBuffer(s_device, renderBuffer->stagingBuffer))
      goto cleanup;

    // Normalize debug AOV heatmaps.
    if (shader_cache->aovId == (int) GiAovId::ClockCycles)
    {
      int valueCount = pixelCount * compCount;

      float max_value = 0.0f;
      for (int i = 0; i < valueCount; i += 4) {
        max_value = std::max(max_value, rgbaImg[i]);
      }
      for (int i = 0; i < valueCount && max_value > 0.0f; i += 4) {
        int val_index = std::min(int((rgbaImg[i] / max_value) * 255.0), 255);
        rgbaImg[i + 0] = (float) TURBO_SRGB_FLOATS[val_index][0];
        rgbaImg[i + 1] = (float) TURBO_SRGB_FLOATS[val_index][1];
        rgbaImg[i + 2] = (float) TURBO_SRGB_FLOATS[val_index][2];
        rgbaImg[i + 3] = 255;
      }
    }

    renderBuffer->sampleOffset += params.spp;

    result = GiStatus::Ok;

cleanup:
    cgpuDestroySemaphore(s_device, semaphore);
    cgpuDestroyCommandBuffer(s_device, commandBuffer);

    return result;
  }

  GiScene* giCreateScene()
  {
    CgpuImage fallbackDomeLightTexture;
    if (!cgpuCreateImage(s_device, { .width = 1, .height = 1 }, &fallbackDomeLightTexture))
    {
      return nullptr;
    }

    GiScene* scene = new GiScene{
      .sphereLights = GgpuDenseDataStore(s_device, *s_stager, *s_delayedResourceDestroyer, sizeof(rp::SphereLight), 64),
      .distantLights = GgpuDenseDataStore(s_device, *s_stager, *s_delayedResourceDestroyer, sizeof(rp::DistantLight), 64),
      .rectLights = GgpuDenseDataStore(s_device, *s_stager, *s_delayedResourceDestroyer, sizeof(rp::RectLight), 64),
      .diskLights = GgpuDenseDataStore(s_device, *s_stager, *s_delayedResourceDestroyer, sizeof(rp::DiskLight), 64),
      .fallbackDomeLightTexture = fallbackDomeLightTexture
    };
    return scene;
  }

  void giDestroyScene(GiScene* scene)
  {
    if (scene->domeLight)
    {
      s_texSys->evictAndDestroyCachedImage(scene->domeLightTexture);
      scene->domeLightTexture.handle = 0;
    }
    cgpuDestroyImage(s_device, scene->fallbackDomeLightTexture);
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

    return light;
  }

  void giDestroySphereLight(GiScene* scene, GiSphereLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->sphereLights.free(light->gpuHandle);
    delete light;
  }

  void giSetSphereLightPosition(GiSphereLight* light, float* pos)
  {
    auto* data = light->scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->pos[0] = pos[0];
    data->pos[1] = pos[1];
    data->pos[2] = pos[2];
  }

  void giSetSphereLightBaseEmission(GiSphereLight* light, float* rgb)
  {
    auto* data = light->scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];
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
  }

  void giSetSphereLightDiffuseSpecular(GiSphereLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->sphereLights.write<rp::SphereLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
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

    return light;
  }

  void giDestroyDistantLight(GiScene* scene, GiDistantLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->distantLights.free(light->gpuHandle);
    delete light;
  }

  void giSetDistantLightDirection(GiDistantLight* light, float* direction)
  {
    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->direction[0] = direction[0];
    data->direction[1] = direction[1];
    data->direction[2] = direction[2];
  }

  void giSetDistantLightBaseEmission(GiDistantLight* light, float* rgb)
  {
    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];
  }

  void giSetDistantLightAngle(GiDistantLight* light, float angle)
  {
    float halfAngle = 0.5f * angle;
    float invPdf = (halfAngle > 0.0f) ? float(2.0f * M_PI * (1.0f - cosf(halfAngle))) : 1.0f;

    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->angle = angle;
    data->invPdf = invPdf;
  }

  void giSetDistantLightDiffuseSpecular(GiDistantLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->distantLights.write<rp::DistantLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
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

    return light;
  }

  void giDestroyRectLight(GiScene* scene, GiRectLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->rectLights.free(light->gpuHandle);
    delete light;
  }

  void giSetRectLightOrigin(GiRectLight* light, float* origin)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->origin[0] = origin[0];
    data->origin[1] = origin[1];
    data->origin[2] = origin[2];
  }

  void giSetRectLightTangents(GiRectLight* light, float* t0, float* t1)
  {
    uint32_t t0packed = _EncodeDirection(glm::make_vec3(t0));
    uint32_t t1packed = _EncodeDirection(glm::make_vec3(t1));

    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->tangentFramePacked = glm::uvec2(t0packed, t1packed);
  }

  void giSetRectLightBaseEmission(GiRectLight* light, float* rgb)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];
  }

  void giSetRectLightDimensions(GiRectLight* light, float width, float height)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->width = width;
    data->height = height;
  }

  void giSetRectLightDiffuseSpecular(GiRectLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->rectLights.write<rp::RectLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
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

    return light;
  }

  void giDestroyDiskLight(GiScene* scene, GiDiskLight* light)
  {
    std::lock_guard guard(scene->mutex);

    scene->diskLights.free(light->gpuHandle);
    delete light;
  }

  void giSetDiskLightOrigin(GiDiskLight* light, float* origin)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->origin[0] = origin[0];
    data->origin[1] = origin[1];
    data->origin[2] = origin[2];
  }

  void giSetDiskLightTangents(GiDiskLight* light, float* t0, float* t1)
  {
    uint32_t t0packed = _EncodeDirection(glm::make_vec3(t0));
    uint32_t t1packed = _EncodeDirection(glm::make_vec3(t1));

    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->tangentFramePacked = glm::uvec2(t0packed, t1packed);
  }

  void giSetDiskLightBaseEmission(GiDiskLight* light, float* rgb)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->baseEmission[0] = rgb[0];
    data->baseEmission[1] = rgb[1];
    data->baseEmission[2] = rgb[2];
  }

  void giSetDiskLightRadius(GiDiskLight* light, float radiusX, float radiusY)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->radiusX = radiusX;
    data->radiusY = radiusY;
  }

  void giSetDiskLightDiffuseSpecular(GiDiskLight* light, float diffuse, float specular)
  {
    auto* data = light->scene->diskLights.write<rp::DiskLight>(light->gpuHandle);
    assert(data);

    data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
  }

  GiDomeLight* giCreateDomeLight(GiScene* scene, const char* filePath)
  {
    std::lock_guard guard(scene->mutex);

    auto* light = new GiDomeLight;
    light->scene = scene;
    light->textureFilePath = filePath;
    return light;
  }

  void giDestroyDomeLight(GiScene* scene, GiDomeLight* light)
  {
    std::lock_guard guard(scene->mutex);
    delete light;
  }

  void giSetDomeLightRotation(GiDomeLight* light, float* quat)
  {
    light->rotation = glm::make_quat(quat);
  }

  void giSetDomeLightBaseEmission(GiDomeLight* light, float* rgb)
  {
    light->baseEmission = glm::make_vec3(rgb);
  }

  void giSetDomeLightDiffuseSpecular(GiDomeLight* light, float diffuse, float specular)
  {
    light->diffuse = diffuse;
    light->specular = specular;
  }

  GiRenderBuffer* giCreateRenderBuffer(uint32_t width, uint32_t height)
  {
    return new GiRenderBuffer{
      .width = width,
      .height = height
    };
  }

  void giDestroyRenderBuffer(GiRenderBuffer* renderBuffer)
  {
    // FIXME: don't destroy resources in use (append them to deletion queue?)

    if (renderBuffer->buffer.handle)
      cgpuDestroyBuffer(s_device, renderBuffer->buffer);
    if (renderBuffer->stagingBuffer.handle)
      cgpuDestroyBuffer(s_device, renderBuffer->stagingBuffer);

    delete renderBuffer;
  }
}
