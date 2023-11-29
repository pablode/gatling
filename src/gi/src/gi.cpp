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

#include "gi.h"

#include "textureManager.h"
#include "turbo.h"
#include "assetReader.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <atomic>
#include <optional>
#include <mutex>
#include <assert.h>

#include <stager.h>
#include <denseDataStore.h>
#include <cgpu.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#ifndef NDEBUG
#include <efsw/efsw.hpp>
#endif
#include "GlslShaderGen.h"
#include <Material.h>
#include <Frontend.h>
#include <Runtime.h>
#include <log.h>
#include "interface/rp_main.h"

#include <MaterialXCore/Document.h>

using namespace gtl;

namespace Rp = gtl::shader_interface::rp_main;

const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

namespace gtl
{
  class McRuntime;
}

struct GiGpuBufferView
{
  uint64_t offset;
  uint64_t size;
};

struct GiGeomCache
{
  std::vector<CgpuBlas> blases;
  CgpuBuffer            buffer;
  GiGpuBufferView       faceBufferView = {};
  CgpuTlas              tlas;
  GiGpuBufferView       vertexBufferView = {};
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
  std::vector<GiFace> faces;
  std::vector<GiVertex> vertices;
  const GiMaterial* material;
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

bool s_loggerInitialized = false;
CgpuDevice s_device;
CgpuPhysicalDeviceFeatures s_deviceFeatures;
CgpuPhysicalDeviceProperties s_deviceProperties;
CgpuSampler s_texSampler;
std::unique_ptr<gtl::GgpuStager> s_stager;
std::unique_ptr<GiGlslShaderGen> s_shaderGen;
std::unique_ptr<McRuntime> s_mcRuntime;
std::unique_ptr<McFrontend> s_mcFrontend;
std::unique_ptr<GiMmapAssetReader> s_mmapAssetReader;
std::unique_ptr<GiAggregateAssetReader> s_aggregateAssetReader;
std::unique_ptr<gtl::GiTextureManager> s_texSys;
std::atomic_bool s_forceShaderCacheInvalid = false;
std::atomic_bool s_forceGeomCacheInvalid = false;
std::atomic_bool s_resetSampleOffset = false;

#ifndef NDEBUG
class ShaderFileListener : public efsw::FileWatchListener
{
public:
  void handleFileAction(efsw::WatchID watchId, const std::string& dir, const std::string& filename,
                        efsw::Action action, std::string oldFilename) override
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
  uint64_t bufferSize = width * height * pixelStride;

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

  {
    CgpuBufferCreateInfo createInfo = {
      .usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
      .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
      .size = bufferSize,
      .debugName = "RenderBuffer"
    };

    if (!cgpuCreateBuffer(s_device, &createInfo, &renderBuffer->buffer))
    {
      return false;
    }
  }

  {
    CgpuBufferCreateInfo createInfo = {
      .usage = CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
      .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
      .size = bufferSize,
      .debugName = "RenderBufferStaging"
    };

    if (!cgpuCreateBuffer(s_device, &createInfo, &renderBuffer->stagingBuffer))
    {
      cgpuDestroyBuffer(s_device, renderBuffer->buffer);
      return false;
    }
  }

  renderBuffer->bufferWidth = width;
  renderBuffer->bufferHeight = height;
  renderBuffer->size = bufferSize;

  return true;
}

void _PrintInitInfo(const GiInitParams* params)
{
  GB_LOG("gatling {}.{}.{} built against MaterialX {}.{}.{}", GATLING_VERSION_MAJOR, GATLING_VERSION_MINOR, GATLING_VERSION_PATCH,
                                                              MATERIALX_MAJOR_VERSION, MATERIALX_MINOR_VERSION, MATERIALX_BUILD_VERSION);
  GB_LOG("> resource path: \"{}\"", params->resourcePath);
  GB_LOG("> shader path: \"{}\"", params->shaderPath);
  GB_LOG("> MDL search paths: {}", params->mdlSearchPaths);
}

GiStatus giInitialize(const GiInitParams* params)
{
  if (!s_loggerInitialized)
  {
    gbLogInit();
    s_loggerInitialized = true;
  }

  _PrintInitInfo(params);

  if (!cgpuInitialize("gatling", GATLING_VERSION_MAJOR, GATLING_VERSION_MINOR, GATLING_VERSION_PATCH))
    return GI_ERROR;

  if (!cgpuCreateDevice(&s_device))
    return GI_ERROR;

  if (!cgpuGetPhysicalDeviceFeatures(s_device, &s_deviceFeatures))
    return GI_ERROR;

  if (!cgpuGetPhysicalDeviceProperties(s_device, &s_deviceProperties))
    return GI_ERROR;

  {
    CgpuSamplerCreateInfo createInfo = {
      .addressModeU = CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
    };

    if (!cgpuCreateSampler(s_device, &createInfo , &s_texSampler))
    {
      return GI_ERROR;
    }
  }

  s_stager = std::make_unique<gtl::GgpuStager>(s_device);
  if (!s_stager->allocate())
  {
    return GI_ERROR;
  }

#ifdef NDEBUG
  const char* shaderPath = params->shaderPath;
#else
  // Use shaders dir in source tree for auto-reloading
  const char* shaderPath = GATLING_SHADER_SOURCE_DIR;
#endif

  s_mcRuntime = std::unique_ptr<McRuntime>(McLoadRuntime(params->resourcePath));
  if (!s_mcRuntime)
  {
    return GI_ERROR;
  }

  MaterialX::DocumentPtr mtlxStdLib = static_pointer_cast<MaterialX::Document>(params->mtlxStdLib);
  if (!mtlxStdLib)
  {
    return GI_ERROR;
  }
  s_mcFrontend = std::make_unique<McFrontend>(params->mdlSearchPaths, mtlxStdLib, *s_mcRuntime);

  s_shaderGen = std::make_unique<GiGlslShaderGen>();
  if (!s_shaderGen->init(shaderPath, *s_mcRuntime))
  {
    return GI_ERROR;
  }

  s_mmapAssetReader = std::make_unique<GiMmapAssetReader>();
  s_aggregateAssetReader = std::make_unique<GiAggregateAssetReader>();
  s_aggregateAssetReader->addAssetReader(s_mmapAssetReader.get());

  s_texSys = std::make_unique<gtl::GiTextureManager>(s_device, *s_aggregateAssetReader, *s_stager);

#ifndef NDEBUG
  s_fileWatcher = std::make_unique<efsw::FileWatcher>();
  s_fileWatcher->addWatch(shaderPath, &s_shaderFileListener, true);
  s_fileWatcher->watch();
#endif

  return GI_OK;
}

void giTerminate()
{
  GB_LOG("terminating...");
#ifndef NDEBUG
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
    s_stager->free();
    s_stager.reset();
  }
  cgpuDestroySampler(s_device, s_texSampler);
  cgpuDestroyDevice(s_device);
  cgpuTerminate();
  s_mcFrontend.reset();
  s_mcRuntime.reset();
}

void giRegisterAssetReader(::GiAssetReader* reader)
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
  MaterialX::DocumentPtr resolvedDoc = static_pointer_cast<MaterialX::Document>(doc);
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

GiMesh* giCreateMesh(const GiMeshDesc* desc)
{
  GiMesh* mesh = new GiMesh;
  mesh->faces = std::vector<GiFace>(&desc->faces[0], &desc->faces[desc->faceCount]);
  mesh->vertices = std::vector<GiVertex>(&desc->vertices[0], &desc->vertices[desc->vertexCount]);
  mesh->material = desc->material;
  return mesh;
}

bool _giBuildGeometryStructures(const GiGeomCacheParams* params,
                                std::vector<CgpuBlas>& blases,
                                std::vector<CgpuBlasInstance>& blasInstances,
                                std::vector<Rp::FVertex>& allVertices,
                                std::vector<Rp::Face>& allFaces)
{
  struct ProtoBlasInstance
  {
    CgpuBlas blas;
    uint32_t faceIndexOffset;
    uint32_t materialIndex;
  };
  std::unordered_map<const GiMesh*, ProtoBlasInstance> protoBlasInstances;

  for (uint32_t m = 0; m < params->meshInstanceCount; m++)
  {
    const GiMeshInstance* instance = &params->meshInstances[m];
    const GiMesh* mesh = instance->mesh;

    if (mesh->faces.empty())
    {
      continue;
    }

    // Build mesh BLAS if it doesn't exist yet.
    if (protoBlasInstances.count(mesh) == 0)
    {
      // Find material for SBT index (FIXME: find a better solution)
      uint32_t materialIndex = UINT32_MAX;
      GiShaderCache* shaderCache = params->shaderCache;
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

      uint32_t faceIndexOffset = allFaces.size();
      uint32_t vertexIndexOffset = allVertices.size();

      // Vertices
      std::vector<CgpuVertex> vertices;
      vertices.resize(mesh->vertices.size());
      allVertices.reserve(allVertices.size() + mesh->vertices.size());

      for (uint32_t i = 0; i < vertices.size(); i++)
      {
        const GiVertex& cpuVert = mesh->vertices[i];

        vertices[i].x = cpuVert.pos[0];
        vertices[i].y = cpuVert.pos[1];
        vertices[i].z = cpuVert.pos[2];

        uint32_t encodedNormal = _EncodeDirection(glm::make_vec3(cpuVert.norm));
        uint32_t encodedTangent = _EncodeDirection(glm::make_vec3(cpuVert.tangent));

        allVertices.push_back(Rp::FVertex{
          .field1 = { glm::make_vec3(cpuVert.pos), cpuVert.bitangentSign },
          .field2 = { *((float*) &encodedNormal), *((float*) &encodedTangent), cpuVert.u, cpuVert.v }
        });
      }

      // Indices
      std::vector<uint32_t> indices;
      indices.reserve(mesh->faces.size() * 3);
      allFaces.reserve(allFaces.size() + mesh->faces.size());

      for (uint32_t i = 0; i < mesh->faces.size(); i++)
      {
        const auto* face = &mesh->faces[i];
        indices.push_back(face->v_i[0]);
        indices.push_back(face->v_i[1]);
        indices.push_back(face->v_i[2]);

        allFaces.push_back(Rp::Face{
          vertexIndexOffset + face->v_i[0],
          vertexIndexOffset + face->v_i[1],
          vertexIndexOffset + face->v_i[2]
        });
      }

      if ((indices.size() % 3) != 0)
      {
        GB_ERROR("BLAS indices do not represent triangles");
        continue;
      }

      // Buffer upload
      CgpuBuffer indexBuffer;
      CgpuBuffer vertexBuffer;

      uint64_t indexBufferSize = indices.size() * sizeof(uint32_t);
      uint64_t vertexBufferSize = vertices.size() * sizeof(CgpuVertex);

      CgpuBufferCreateInfo iboCreateInfo = {
        .usage = CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
        .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
        .size = indexBufferSize,
        .debugName = "BlasIndices"
      };

      if (!cgpuCreateBuffer(s_device, &iboCreateInfo, &indexBuffer))
      {
        GB_ERROR("failed to allocate BLAS indices memory");
        continue;
      }

      CgpuBufferCreateInfo vboCreateInfo = {
        .usage = CGPU_BUFFER_USAGE_FLAG_SHADER_DEVICE_ADDRESS | CGPU_BUFFER_USAGE_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT,
        .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
        .size = vertexBufferSize,
        .debugName = "BlasVertices"
      };

      if (!cgpuCreateBuffer(s_device, &vboCreateInfo, &vertexBuffer))
      {
        cgpuDestroyBuffer(s_device, vertexBuffer);
        GB_ERROR("failed to allocate BLAS vertices memory");
        continue;
      }

      void* mappedMem;
      cgpuMapBuffer(s_device, indexBuffer, &mappedMem);
      memcpy(mappedMem, indices.data(), indexBufferSize);
      cgpuUnmapBuffer(s_device, indexBuffer);

      cgpuMapBuffer(s_device, vertexBuffer, &mappedMem);
      memcpy(mappedMem, vertices.data(), vertexBufferSize);
      cgpuUnmapBuffer(s_device, vertexBuffer);

      // BLAS
      CgpuBlasCreateInfo blasCreateInfo = {
        .vertexBuffer = vertexBuffer,
        .indexBuffer = indexBuffer,
        .maxVertex = (uint32_t) vertices.size(),
        .triangleCount = (uint32_t) indices.size() / 3,
        .isOpaque = mesh->material->mcMat->isOpaque
      };

      CgpuBlas blas;
      bool blasCreated = cgpuCreateBlas(s_device, &blasCreateInfo, &blas);

      cgpuDestroyBuffer(s_device, indexBuffer);
      cgpuDestroyBuffer(s_device, vertexBuffer);

      if (!blasCreated)
      {
        GB_ERROR("failed to allocate BLAS vertex memory");
        continue;
      }

      blases.push_back(blas);

      ProtoBlasInstance proto;
      proto.blas = blas;
      proto.faceIndexOffset = faceIndexOffset;
      proto.materialIndex = materialIndex;
      protoBlasInstances[mesh] = proto;
    }

    // Create mesh instance for TLAS.
    const ProtoBlasInstance& proto = protoBlasInstances[mesh];

    CgpuBlasInstance blasInstance;
    blasInstance.as = proto.blas;
    blasInstance.faceIndexOffset = proto.faceIndexOffset;
    blasInstance.hitGroupIndex = proto.materialIndex * 2; // always two hit groups per material: regular & shadow
    memcpy(blasInstance.transform, instance->transform, sizeof(float) * 12);

    blasInstances.push_back(blasInstance);
  }

  return true;

fail_cleanup:
  assert(false);
  for (CgpuBlas blas : blases)
  {
    cgpuDestroyBlas(s_device, blas);
  }
  return false;
}

GiGeomCache* giCreateGeomCache(const GiGeomCacheParams* params)
{
  s_forceGeomCacheInvalid = false;

  GiGeomCache* cache = nullptr;

  GB_LOG("instance count: {}", params->meshInstanceCount);
  GB_LOG("creating geom cache..");
  fflush(stdout);

  // Build HW ASes and vertex, index buffers.
  CgpuBuffer buffer;
  CgpuTlas tlas;
  std::vector<CgpuBlas> blases;
  std::vector<CgpuBlasInstance> blasInstances;
  std::vector<Rp::FVertex> allVertices;
  std::vector<Rp::Face> allFaces;

  if (!_giBuildGeometryStructures(params, blases, blasInstances, allVertices, allFaces))
    goto cleanup;

  {
    CgpuTlasCreateInfo tlasCreateInfo = {
      .instanceCount = (uint32_t) blasInstances.size(),
      .instances = blasInstances.data()
    };

    if (!cgpuCreateTlas(s_device, &tlasCreateInfo, &tlas))
      goto cleanup;
  }

  // Upload attribute buffer to GPU.
  GiGpuBufferView faceBufferView;
  GiGpuBufferView vertexBufferView;
  {
    uint64_t attributeBufferSize = 0;
    const uint64_t offset_align = s_deviceProperties.minStorageBufferOffsetAlignment;

    faceBufferView.size = allFaces.size() * sizeof(Rp::Face);
    vertexBufferView.size = allVertices.size() * sizeof(Rp::FVertex);

    faceBufferView.offset = giAlignBuffer(offset_align, faceBufferView.size, &attributeBufferSize);
    vertexBufferView.offset = giAlignBuffer(offset_align, vertexBufferView.size, &attributeBufferSize);

    GB_LOG("total attribute buffer size: {:.2f} MiB", attributeBufferSize * BYTES_TO_MIB);
    GB_LOG("> {:.2f} MiB faces", faceBufferView.size * BYTES_TO_MIB);
    GB_LOG("> {:.2f} MiB vertices", vertexBufferView.size * BYTES_TO_MIB);
    fflush(stdout);

    CgpuBufferCreateInfo createInfo = {
      .usage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
      .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
      .size = attributeBufferSize,
      .debugName = "AsAttributes"
    };

    if (!cgpuCreateBuffer(s_device, &createInfo, &buffer))
      goto cleanup;

    if (!s_stager->stageToBuffer((uint8_t*)allFaces.data(), faceBufferView.size, buffer, faceBufferView.offset))
      goto cleanup;
    if (!s_stager->stageToBuffer((uint8_t*)allVertices.data(), vertexBufferView.size, buffer, vertexBufferView.offset))
      goto cleanup;
  }

  // Fill cache struct.
  cache = new GiGeomCache;
  cache->tlas = tlas;
  cache->blases = blases;
  cache->buffer = buffer;
  cache->faceBufferView = faceBufferView;
  cache->vertexBufferView = vertexBufferView;

cleanup:
  if (!cache)
  {
    assert(false);
    if (buffer.handle)
    {
      cgpuDestroyBuffer(s_device, buffer);
    }
    if (tlas.handle)
    {
      cgpuDestroyTlas(s_device, tlas);
    }
    for (CgpuBlas blas : blases)
    {
      cgpuDestroyBlas(s_device, blas);
    }
  }
  return cache;
}

void giDestroyGeomCache(GiGeomCache* cache)
{
  for (CgpuBlas blas : cache->blases)
  {
    cgpuDestroyBlas(s_device, blas);
  }
  cgpuDestroyTlas(s_device, cache->tlas);
  cgpuDestroyBuffer(s_device, cache->buffer);
  delete cache;
}

// FIXME: move this into the GiScene struct - also, want to rebuild with cached data at shader granularity
bool giShaderCacheNeedsRebuild()
{
  return s_forceShaderCacheInvalid;
}
bool giGeomCacheNeedsRebuild()
{
  return s_forceGeomCacheInvalid;
}

GiShaderCache* giCreateShaderCache(const GiShaderCacheParams* params)
{
  s_forceShaderCacheInvalid = false;

  bool clockCyclesAov = params->aovId == GI_AOV_ID_DEBUG_CLOCK_CYCLES;

  if (clockCyclesAov && !s_deviceFeatures.shaderClock)
  {
    GB_ERROR("unsupported AOV - device feature missing");
    return nullptr;
  }

  GiScene* scene = params->scene;

  GB_LOG("material count: {}", params->materialCount);
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
  uint32_t texCount2d = 2; // +1 fallback and +1 real dome light
  uint32_t texCount3d = 0;
  bool hasPipelineClosestHitShader = false;
  bool hasPipelineAnyHitShader = false;

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
    hitGroupCompInfos.resize(params->materialCount);

    std::atomic_bool threadWorkFailed = false;
#pragma omp parallel for
    for (int i = 0; i < hitGroupCompInfos.size(); i++)
    {
      const McMaterial* material = params->materials[i]->mcMat;

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
      if (!material->isOpaque)
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
    for (int i = 0; i < hitGroupCompInfos.size(); i++)
    {
      const McMaterial* material = params->materials[i]->mcMat;

      HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

      // Closest hit
      {
        GiGlslShaderGen::ClosestHitShaderParams hitParams;
        hitParams.aovId = params->aovId;
        hitParams.baseFileName = "rp_main.chit";
        hitParams.isOpaque = material->isOpaque;
        hitParams.enableSceneTransforms = material->requiresSceneTransforms;
        hitParams.nextEventEstimation = params->nextEventEstimation;
        hitParams.shadingGlsl = compInfo.closestHitInfo.genInfo.glslSource;
        hitParams.sphereLightCount = scene->sphereLights.elementCount();
        hitParams.distantLightCount = scene->distantLights.elementCount();
        hitParams.rectLightCount = scene->rectLights.elementCount();
        hitParams.diskLightCount = scene->diskLights.elementCount();
        hitParams.textureIndexOffset2d = compInfo.closestHitInfo.texOffset2d;
        hitParams.textureIndexOffset3d = compInfo.closestHitInfo.texOffset3d;
        hitParams.texCount2d = texCount2d;
        hitParams.texCount3d = texCount3d;

        if (!s_shaderGen->generateClosestHitSpirv(hitParams, compInfo.closestHitInfo.spv))
        {
          threadWorkFailed = true;
          continue;
        }
      }

      // Any hit
      if (compInfo.anyHitInfo)
      {
        GiGlslShaderGen::AnyHitShaderParams hitParams;
        hitParams.aovId = params->aovId;
        hitParams.enableSceneTransforms = material->requiresSceneTransforms;
        hitParams.baseFileName = "rp_main.ahit";
        hitParams.opacityEvalGlsl = compInfo.anyHitInfo->genInfo.glslSource;
        hitParams.sphereLightCount = scene->sphereLights.elementCount();
        hitParams.distantLightCount = scene->distantLights.elementCount();
        hitParams.rectLightCount = scene->rectLights.elementCount();
        hitParams.diskLightCount = scene->diskLights.elementCount();
        hitParams.textureIndexOffset2d = compInfo.anyHitInfo->texOffset2d;
        hitParams.textureIndexOffset3d = compInfo.anyHitInfo->texOffset3d;
        hitParams.texCount2d = texCount2d;
        hitParams.texCount3d = texCount3d;

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

    for (int i = 0; i < hitGroupCompInfos.size(); i++)
    {
      const HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

      // regular hit group
      {
        CgpuShader closestHitShader;
        {
          const std::vector<uint8_t>& spv = compInfo.closestHitInfo.spv;

          CgpuShaderCreateInfo createInfo = {
            .size = spv.size(),
            .source = spv.data(),
            .stageFlags = CGPU_SHADER_STAGE_FLAG_CLOSEST_HIT
          };

          if (!cgpuCreateShader(s_device, &createInfo, &closestHitShader))
          {
            goto cleanup;
          }

          hitShaders.push_back(closestHitShader);
        }

        CgpuShader anyHitShader;
        if (compInfo.anyHitInfo)
        {
          const std::vector<uint8_t>& spv = compInfo.anyHitInfo->spv;

          CgpuShaderCreateInfo createInfo = {
            .size = spv.size(),
            .source = spv.data(),
            .stageFlags = CGPU_SHADER_STAGE_FLAG_ANY_HIT
          };

          if (!cgpuCreateShader(s_device, &createInfo, &anyHitShader))
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

          CgpuShaderCreateInfo createInfo = {
            .size = spv.size(),
            .source = spv.data(),
            .stageFlags = CGPU_SHADER_STAGE_FLAG_ANY_HIT
          };

          if (!cgpuCreateShader(s_device, &createInfo, &anyHitShader))
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
    GiGlslShaderGen::RaygenShaderParams rgenParams;
    rgenParams.aovId = params->aovId;
    rgenParams.depthOfField = params->depthOfField;
    rgenParams.filterImportanceSampling = params->filterImportanceSampling;
    rgenParams.materialCount = params->materialCount;
    rgenParams.nextEventEstimation = params->nextEventEstimation;
    rgenParams.progressiveAccumulation = params->progressiveAccumulation;
    rgenParams.reorderInvocations = s_deviceFeatures.rayTracingInvocationReorder;
    rgenParams.sphereLightCount = scene->sphereLights.elementCount();
    rgenParams.distantLightCount = scene->distantLights.elementCount();
    rgenParams.rectLightCount = scene->rectLights.elementCount();
    rgenParams.diskLightCount = scene->diskLights.elementCount();
    rgenParams.shaderClockExts = clockCyclesAov;
    rgenParams.texCount2d = texCount2d;
    rgenParams.texCount3d = texCount3d;

    std::vector<uint8_t> spv;
    if (!s_shaderGen->generateRgenSpirv("rp_main.rgen", rgenParams, spv))
    {
      goto cleanup;
    }

    CgpuShaderCreateInfo createInfo = {
      .size = spv.size(),
      .source = spv.data(),
      .stageFlags = CGPU_SHADER_STAGE_FLAG_RAYGEN
    };

    if (!cgpuCreateShader(s_device, &createInfo, &rgenShader))
    {
      goto cleanup;
    }
  }

  // Create miss shaders.
  {
    GiGlslShaderGen::MissShaderParams missParams;
    missParams.domeLightCameraVisible = params->domeLightCameraVisible;
    missParams.sphereLightCount = scene->sphereLights.elementCount();
    missParams.distantLightCount = scene->distantLights.elementCount();
    missParams.rectLightCount = scene->rectLights.elementCount();
    missParams.diskLightCount = scene->diskLights.elementCount();
    missParams.texCount2d = texCount2d;
    missParams.texCount3d = texCount3d;

    // regular miss shader
    {
      std::vector<uint8_t> spv;
      if (!s_shaderGen->generateMissSpirv("rp_main.miss", missParams, spv))
      {
        goto cleanup;
      }

      CgpuShaderCreateInfo createInfo = {
        .size = spv.size(),
        .source = spv.data(),
        .stageFlags = CGPU_SHADER_STAGE_FLAG_MISS
      };

      CgpuShader missShader;
      if (!cgpuCreateShader(s_device, &createInfo, &missShader))
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

      CgpuShaderCreateInfo createInfo = {
        .size = spv.size(),
        .source = spv.data(),
        .stageFlags = CGPU_SHADER_STAGE_FLAG_MISS
      };

      CgpuShader missShader;
      if (!cgpuCreateShader(s_device, &createInfo, &missShader))
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

    CgpuRtPipelineCreateInfo pipelineDesc = {
      .rgenShader = rgenShader,
      .missShaderCount = (uint32_t) missShaders.size(),
      .missShaders = missShaders.data(),
      .hitGroupCount = (uint32_t) hitGroups.size(),
      .hitGroups = hitGroups.data(),
    };

    if (!cgpuCreateRtPipeline(s_device, &pipelineDesc, &pipeline))
    {
      goto cleanup;
    }
  }

  cache = new GiShaderCache;
  cache->aovId = params->aovId;
  cache->domeLightCameraVisible = params->domeLightCameraVisible;
  cache->hitShaders = std::move(hitShaders);
  cache->images2d = std::move(images2d);
  cache->images3d = std::move(images3d);
  cache->materials.resize(params->materialCount);
  for (int i = 0; i < params->materialCount; i++)
  {
    cache->materials[i] = params->materials[i];
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

int giRender(const GiRenderParams* params, float* rgbaImg)
{
  s_stager->flush();

  const GiGeomCache* geom_cache = params->geomCache;
  const GiShaderCache* shader_cache = params->shaderCache;
  GiScene* scene = params->scene;

  // Upload dome lights.
  glm::vec4 backgroundColor = glm::make_vec4(params->backgroundColor);
  if (backgroundColor != scene->backgroundColor)
  {
    glm::u8vec4 u8BgColor(backgroundColor * 255.0f);
    s_stager->stageToImage(glm::value_ptr(u8BgColor), 4, scene->fallbackDomeLightTexture, 1, 1);
    scene->backgroundColor = backgroundColor;
  }

  if (scene->domeLight != params->domeLight)
  {
    if (scene->domeLightTexture.handle &&
        scene->domeLightTexture.handle != scene->fallbackDomeLightTexture.handle)
    {
      s_texSys->evictAndDestroyCachedImage(scene->domeLightTexture);
      scene->domeLightTexture.handle = 0;
    }
    scene->domeLight = nullptr;

    GiDomeLight* domeLight = params->domeLight;
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
  int result = GI_ERROR;

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
  GiRenderBuffer* renderBuffer = params->renderBuffer;
  uint32_t imageWidth = renderBuffer->width;
  uint32_t imageHeight = renderBuffer->height;

  int compCount = 4;
  int pixelStride = compCount * sizeof(float);
  int pixelCount = imageWidth * imageHeight;

  if (!_giResizeRenderBufferIfNeeded(renderBuffer, pixelStride))
  {
    GB_ERROR("failed to resize render buffer!");
    return GI_ERROR;
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

  auto camForward = glm::normalize(glm::make_vec3(params->camera->forward));
  auto camUp = glm::normalize(glm::make_vec3(params->camera->up));

  float lensRadius = 0.0f;
  if (params->camera->fStop > 0.0f)
  {
    lensRadius = params->camera->focalLength / (2.0f * params->camera->fStop);
  }

  glm::quat domeLightRotation = scene->domeLight ? scene->domeLight->rotation : glm::quat()/* doesn't matter, uniform color */;
  glm::vec3 domeLightEmissionMultiplier = scene->domeLight ? scene->domeLight->baseEmission : glm::vec3(1.0f);
  uint32_t domeLightDiffuseSpecularPacked = glm::packHalf2x16(scene->domeLight ? glm::vec2(scene->domeLight->diffuse, scene->domeLight->specular) : glm::vec2(1.0f));

  Rp::PushConstants pushData = {
    .cameraPosition                 = glm::make_vec3(params->camera->position),
    .imageDims                      = ((imageHeight << 16) | imageWidth),
    .cameraForward                  = camForward,
    .focusDistance                  = params->camera->focusDistance,
    .cameraUp                       = camUp,
    .cameraVFoV                     = params->camera->vfov,
    .sampleOffset                   = renderBuffer->sampleOffset,
    .lensRadius                     = lensRadius,
    .sampleCount                    = params->spp,
    .maxSampleValue                 = params->maxSampleValue,
    .domeLightRotation              = glm::make_vec4(&domeLightRotation[0]),
    .domeLightEmissionMultiplier    = domeLightEmissionMultiplier,
    .domeLightDiffuseSpecularPacked = domeLightDiffuseSpecularPacked,
    .maxBouncesAndRrBounceOffset    = ((params->maxBounces << 16) | params->rrBounceOffset),
    .rrInvMinTermProb               = params->rrInvMinTermProb,
    .lightIntensityMultiplier       = params->lightIntensityMultiplier,
    .clipRangePacked                = glm::packHalf2x16(glm::vec2(params->camera->clipStart, params->camera->clipEnd))
  };

  std::vector<CgpuBufferBinding> buffers;
  buffers.reserve(16);

  buffers.push_back({ .binding = Rp::BINDING_INDEX_OUT_PIXELS, .buffer = renderBuffer->buffer });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_FACES, .buffer = geom_cache->buffer,
                      .offset = geom_cache->faceBufferView.offset, .size = geom_cache->faceBufferView.size });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_VERTICES, .buffer = geom_cache->buffer,
                      .offset = geom_cache->vertexBufferView.offset, .size = geom_cache->vertexBufferView.size });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_SPHERE_LIGHTS, .buffer = scene->sphereLights.buffer() });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_DISTANT_LIGHTS, .buffer = scene->distantLights.buffer() });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_RECT_LIGHTS, .buffer = scene->rectLights.buffer() });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_DISK_LIGHTS, .buffer = scene->diskLights.buffer() });

  size_t imageCount = shader_cache->images2d.size() + shader_cache->images3d.size() + 2/* dome lights */;

  std::vector<CgpuImageBinding> images;
  images.reserve(imageCount);

  CgpuSamplerBinding sampler = { .binding = Rp::BINDING_INDEX_SAMPLER, .sampler = s_texSampler };

  images.push_back({ .binding = Rp::BINDING_INDEX_TEXTURES_2D, .image = scene->fallbackDomeLightTexture, .index = 0 });
  images.push_back({ .binding = Rp::BINDING_INDEX_TEXTURES_2D, .image = scene->domeLightTexture,         .index = 1 });

  for (uint32_t i = 0; i < shader_cache->images2d.size(); i++)
  {
    images.push_back({ .binding = Rp::BINDING_INDEX_TEXTURES_2D,
                       .image = shader_cache->images2d[i],
                       .index = 2/* dome lights */ + i });
  }
  for (uint32_t i = 0; i < shader_cache->images3d.size(); i++)
  {
    images.push_back({ .binding = Rp::BINDING_INDEX_TEXTURES_3D, .image = shader_cache->images3d[i], .index = i });
  }

  CgpuTlasBinding as = { .binding = Rp::BINDING_INDEX_SCENE_AS, .as = geom_cache->tlas };

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

  if (!cgpuCmdTransitionShaderImageLayouts(commandBuffer, shader_cache->rgenShader, images.size(), images.data()))
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

  // Read data from GPU to image.
  uint8_t* mapped_staging_mem;
  if (!cgpuMapBuffer(s_device, renderBuffer->stagingBuffer, (void**) &mapped_staging_mem))
    goto cleanup;

  memcpy(rgbaImg, mapped_staging_mem, renderBuffer->size);

  if (!cgpuUnmapBuffer(s_device, renderBuffer->stagingBuffer))
    goto cleanup;

  // Normalize debug AOV heatmaps.
  if (shader_cache->aovId == GI_AOV_ID_DEBUG_CLOCK_CYCLES)
  {
    int valueCount = pixelCount * compCount;

    float max_value = 0.0f;
    for (int i = 0; i < valueCount; i += 4) {
      max_value = std::max(max_value, rgbaImg[i]);
    }
    for (int i = 0; i < valueCount && max_value > 0.0f; i += 4) {
      int val_index = std::min(int((rgbaImg[i] / max_value) * 255.0), 255);
      rgbaImg[i + 0] = (float) gtl::TURBO_SRGB_FLOATS[val_index][0];
      rgbaImg[i + 1] = (float) gtl::TURBO_SRGB_FLOATS[val_index][1];
      rgbaImg[i + 2] = (float) gtl::TURBO_SRGB_FLOATS[val_index][2];
      rgbaImg[i + 3] = 255;
    }
  }

  renderBuffer->sampleOffset += params->spp;

  result = GI_OK;

cleanup:
  cgpuDestroySemaphore(s_device, semaphore);
  cgpuDestroyCommandBuffer(s_device, commandBuffer);

  return result;
}

GiScene* giCreateScene()
{
  CgpuImageCreateInfo imgCreateInfo = { .width = 1, .height = 1 };
  CgpuImage fallbackDomeLightTexture;
  if (!cgpuCreateImage(s_device, &imgCreateInfo, &fallbackDomeLightTexture))
  {
    return nullptr;
  }

  GiScene* scene = new GiScene{
    .sphereLights = GgpuDenseDataStore(s_device, *s_stager, sizeof(Rp::SphereLight), 64),
    .distantLights = GgpuDenseDataStore(s_device, *s_stager, sizeof(Rp::DistantLight), 64),
    .rectLights = GgpuDenseDataStore(s_device, *s_stager, sizeof(Rp::RectLight), 64),
    .diskLights = GgpuDenseDataStore(s_device, *s_stager, sizeof(Rp::DiskLight), 64),
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
  auto* light = new GiSphereLight;
  light->scene = scene;
  light->gpuHandle = scene->sphereLights.allocate();

  auto* data = scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
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
  scene->sphereLights.free(light->gpuHandle);
  delete light;
}

void giSetSphereLightPosition(GiSphereLight* light, float* pos)
{
  auto* data = light->scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
  assert(data);

  data->pos[0] = pos[0];
  data->pos[1] = pos[1];
  data->pos[2] = pos[2];
}

void giSetSphereLightBaseEmission(GiSphereLight* light, float* rgb)
{
  auto* data = light->scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
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
  float area = powf((ab + ac + bc) / 3.0f, 1.0f / 1.6f) * 4.0f * M_PI;

  auto* data = light->scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
  assert(data);

  data->radiusXYZ[0] = radiusX;
  data->radiusXYZ[1] = radiusY;
  data->radiusXYZ[2] = radiusZ;
  data->area = area;
}

void giSetSphereLightDiffuseSpecular(GiSphereLight* light, float diffuse, float specular)
{
  auto* data = light->scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
  assert(data);

  data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
}

GiDistantLight* giCreateDistantLight(GiScene* scene)
{
  auto* light = new GiDistantLight;
  light->scene = scene;
  light->gpuHandle = scene->distantLights.allocate();

  auto* data = scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
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
  scene->distantLights.free(light->gpuHandle);
  delete light;
}

void giSetDistantLightDirection(GiDistantLight* light, float* direction)
{
  auto* data = light->scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->direction[0] = direction[0];
  data->direction[1] = direction[1];
  data->direction[2] = direction[2];
}

void giSetDistantLightBaseEmission(GiDistantLight* light, float* rgb)
{
  auto* data = light->scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->baseEmission[0] = rgb[0];
  data->baseEmission[1] = rgb[1];
  data->baseEmission[2] = rgb[2];
}

void giSetDistantLightAngle(GiDistantLight* light, float angle)
{
  float halfAngle = 0.5 * angle;
  float invPdf = (halfAngle > 0.0f) ? (2.0f * M_PI * (1.0f - cosf(halfAngle))) : 1.0f;

  auto* data = light->scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->angle = angle;
  data->invPdf = invPdf;
}

void giSetDistantLightDiffuseSpecular(GiDistantLight* light, float diffuse, float specular)
{
  auto* data = light->scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
}

GiRectLight* giCreateRectLight(GiScene* scene)
{
  auto* light = new GiRectLight;
  light->scene = scene;
  light->gpuHandle = scene->rectLights.allocate();

  uint32_t t0packed = _EncodeDirection(glm::vec3(1.0f, 0.0f, 0.0f));
  uint32_t t1packed = _EncodeDirection(glm::vec3(0.0f, 1.0f, 0.0f));

  auto* data = scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
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
  scene->rectLights.free(light->gpuHandle);
  delete light;
}

void giSetRectLightOrigin(GiRectLight* light, float* origin)
{
  auto* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->origin[0] = origin[0];
  data->origin[1] = origin[1];
  data->origin[2] = origin[2];
}

void giSetRectLightTangents(GiRectLight* light, float* t0, float* t1)
{
  uint32_t t0packed = _EncodeDirection(glm::make_vec3(t0));
  uint32_t t1packed = _EncodeDirection(glm::make_vec3(t1));

  auto* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->tangentFramePacked = glm::uvec2(t0packed, t1packed);
}

void giSetRectLightBaseEmission(GiRectLight* light, float* rgb)
{
  auto* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->baseEmission[0] = rgb[0];
  data->baseEmission[1] = rgb[1];
  data->baseEmission[2] = rgb[2];
}

void giSetRectLightDimensions(GiRectLight* light, float width, float height)
{
  auto* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->width = width;
  data->height = height;
}

void giSetRectLightDiffuseSpecular(GiRectLight* light, float diffuse, float specular)
{
  auto* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
}

GiDiskLight* giCreateDiskLight(GiScene* scene)
{
  auto* light = new GiDiskLight;
  light->scene = scene;
  light->gpuHandle = scene->diskLights.allocate();

  uint32_t t0packed = _EncodeDirection(glm::vec3(1.0f, 0.0f, 0.0f));
  uint32_t t1packed = _EncodeDirection(glm::vec3(0.0f, 1.0f, 0.0f));

  auto* data = scene->diskLights.write<Rp::DiskLight>(light->gpuHandle);
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
  scene->diskLights.free(light->gpuHandle);
  delete light;
}

void giSetDiskLightOrigin(GiDiskLight* light, float* origin)
{
  auto* data = light->scene->diskLights.write<Rp::DiskLight>(light->gpuHandle);
  assert(data);

  data->origin[0] = origin[0];
  data->origin[1] = origin[1];
  data->origin[2] = origin[2];
}

void giSetDiskLightTangents(GiDiskLight* light, float* t0, float* t1)
{
  uint32_t t0packed = _EncodeDirection(glm::make_vec3(t0));
  uint32_t t1packed = _EncodeDirection(glm::make_vec3(t1));

  auto* data = light->scene->diskLights.write<Rp::DiskLight>(light->gpuHandle);
  assert(data);

  data->tangentFramePacked = glm::uvec2(t0packed, t1packed);
}

void giSetDiskLightBaseEmission(GiDiskLight* light, float* rgb)
{
  auto* data = light->scene->diskLights.write<Rp::DiskLight>(light->gpuHandle);
  assert(data);

  data->baseEmission[0] = rgb[0];
  data->baseEmission[1] = rgb[1];
  data->baseEmission[2] = rgb[2];
}

void giSetDiskLightRadius(GiDiskLight* light, float radiusX, float radiusY)
{
  auto* data = light->scene->diskLights.write<Rp::DiskLight>(light->gpuHandle);
  assert(data);

  data->radiusX = radiusX;
  data->radiusY = radiusY;
}

void giSetDiskLightDiffuseSpecular(GiDiskLight* light, float diffuse, float specular)
{
  auto* data = light->scene->diskLights.write<Rp::DiskLight>(light->gpuHandle);
  assert(data);

  data->diffuseSpecularPacked = glm::packHalf2x16(glm::vec2(diffuse, specular));
}

GiDomeLight* giCreateDomeLight(GiScene* scene, const char* filePath)
{
  auto* light = new GiDomeLight;
  light->scene = scene;
  light->textureFilePath = filePath;
  return light;
}

void giDestroyDomeLight(GiScene* scene, GiDomeLight* light)
{
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
