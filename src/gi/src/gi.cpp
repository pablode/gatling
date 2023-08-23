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

#include "texsys.h"
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
#include "sg/ShaderGen.h"
#include "interface/rp_main.h"

#include <MaterialXCore/Document.h>

using namespace gi;
using namespace gtl;

namespace Rp = gtl::shader_interface::rp_main;

const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

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
  std::vector<CgpuShader>        hitShaders;
  std::vector<CgpuImage>         images2d;
  std::vector<CgpuImage>         images3d;
  std::vector<const GiMaterial*> materials;
  std::vector<CgpuShader>        missShaders;
  CgpuPipeline                   pipeline;
  bool                           hasPipelineClosestHitShader = false;
  bool                           hasPipelineAnyHitShader = false;
  CgpuShader                     rgenShader;
};

struct GiMaterial
{
  sg::Material* sgMat;
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

struct GiScene
{
  GgpuDenseDataStore sphereLights;
  GgpuDenseDataStore distantLights;
  GgpuDenseDataStore rectLights;
  CgpuImage domeLightTexture;
  glm::mat3 domeLightTransform;
  GiDomeLight* domeLight; // weak ptr
};

struct GiDomeLight
{
  std::string textureFilePath;
  glm::mat3 transform{1.0f};
};

CgpuDevice s_device;
CgpuPhysicalDeviceFeatures s_deviceFeatures;
CgpuPhysicalDeviceProperties s_deviceProperties;
CgpuSampler s_texSampler;
std::unique_ptr<gtl::GgpuStager> s_stager;
std::unique_ptr<sg::ShaderGen> s_shaderGen;
std::unique_ptr<GiMmapAssetReader> s_mmapAssetReader;
std::unique_ptr<GiAggregateAssetReader> s_aggregateAssetReader;
std::unique_ptr<gi::TexSys> s_texSys;
CgpuBuffer s_outputBuffer;
CgpuBuffer s_outputStagingBuffer;
uint32_t s_outputBufferWidth = 0;
uint32_t s_outputBufferHeight = 0;
uint32_t s_sampleOffset = 0;
std::atomic_bool s_forceShaderCacheInvalid = false;
std::atomic_bool s_forceGeomCacheInvalid = false;

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
      s_sampleOffset = 0;
      break;
    default:
      break;
    }
  }
};

std::unique_ptr<efsw::FileWatcher> s_fileWatcher;
ShaderFileListener s_shaderFileListener;
#endif

bool _giResizeOutputBuffer(uint32_t width, uint32_t height, uint32_t bufferSize)
{
  s_outputBufferWidth = width;
  s_outputBufferHeight = height;

  if (s_outputBuffer.handle)
  {
    cgpuDestroyBuffer(s_device, s_outputBuffer);
    s_outputBuffer.handle = 0;
  }
  if (s_outputStagingBuffer.handle)
  {
    cgpuDestroyBuffer(s_device, s_outputStagingBuffer);
    s_outputStagingBuffer.handle = 0;
  }

  if (width == 0 || height == 0)
  {
    return true;
  }

  if (!cgpuCreateBuffer(s_device,
                        CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
                        CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                        bufferSize,
                        &s_outputBuffer))
  {
    return false;
  }

  if (!cgpuCreateBuffer(s_device,
                        CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
                        CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
                        bufferSize,
                        &s_outputStagingBuffer))
  {
    return false;
  }

  return true;
}

GiStatus giInitialize(const GiInitParams* params)
{
  if (!cgpuInitialize("gatling", GATLING_VERSION_MAJOR, GATLING_VERSION_MINOR, GATLING_VERSION_PATCH))
    return GI_ERROR;

  if (!cgpuCreateDevice(&s_device))
    return GI_ERROR;

  if (!cgpuGetPhysicalDeviceFeatures(s_device, &s_deviceFeatures))
    return GI_ERROR;

  if (!cgpuGetPhysicalDeviceProperties(s_device, &s_deviceProperties))
    return GI_ERROR;

  if (!cgpuCreateSampler(s_device,
                           CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                           CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                           CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                           &s_texSampler))
  {
    return GI_ERROR;
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

  sg::ShaderGen::InitParams sgParams = {
    .resourcePath = params->resourcePath,
    .shaderPath = shaderPath,
    .mdlSearchPaths = params->mdlSearchPaths,
    .mtlxSearchPaths = params->mtlxSearchPaths
  };

  s_shaderGen = std::make_unique<sg::ShaderGen>();
  if (!s_shaderGen->init(sgParams))
  {
    return GI_ERROR;
  }

  s_mmapAssetReader = std::make_unique<GiMmapAssetReader>();
  s_aggregateAssetReader = std::make_unique<GiAggregateAssetReader>();
  s_aggregateAssetReader->addAssetReader(s_mmapAssetReader.get());

  s_texSys = std::make_unique<gi::TexSys>(s_device, *s_aggregateAssetReader, *s_stager);

#ifndef NDEBUG
  s_fileWatcher = std::make_unique<efsw::FileWatcher>();
  s_fileWatcher->addWatch(shaderPath, &s_shaderFileListener, true);
  s_fileWatcher->watch();
#endif

  return GI_OK;
}

void giTerminate()
{
#ifndef NDEBUG
  s_fileWatcher.reset();
#endif
  s_aggregateAssetReader.reset();
  s_mmapAssetReader.reset();
  _giResizeOutputBuffer(0, 0, 0);
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
}

void giRegisterAssetReader(GiAssetReader* reader)
{
  s_aggregateAssetReader->addAssetReader(reader);
}

GiMaterial* giCreateMaterialFromMtlxStr(const char* str)
{
  sg::Material* sgMat = s_shaderGen->createMaterialFromMtlxStr(str);
  if (!sgMat)
  {
    return nullptr;
  }

  GiMaterial* mat = new GiMaterial;
  mat->sgMat = sgMat;
  return mat;
}

GiMaterial* giCreateMaterialFromMtlxDoc(const std::shared_ptr<void/*MaterialX::Document*/> doc)
{
  MaterialX::DocumentPtr resolvedDoc = static_pointer_cast<MaterialX::Document>(doc);
  if (!doc)
  {
    return nullptr;
  }

  sg::Material* sgMat = s_shaderGen->createMaterialFromMtlxDoc(resolvedDoc);
  if (!sgMat)
  {
    return nullptr;
  }

  GiMaterial* mat = new GiMaterial;
  mat->sgMat = sgMat;
  return mat;
}

GiMaterial* giCreateMaterialFromMdlFile(const char* filePath, const char* subIdentifier)
{
  sg::Material* sgMat = s_shaderGen->createMaterialFromMdlFile(filePath, subIdentifier);
  if (!sgMat)
  {
    return nullptr;
  }

  GiMaterial* mat = new GiMaterial;
  mat->sgMat = sgMat;
  return mat;
}

void giDestroyMaterial(GiMaterial* mat)
{
  s_shaderGen->destroyMaterial(mat->sgMat);
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

        const auto encodeOctahedral = [](glm::vec3 v) {
          v /= (fabsf(v.x) + fabsf(v.y) + fabsf(v.z));
          glm::vec2 ps = glm::vec2(v.x >= 0.0f ? +1.0f : -1.0f, v.y >= 0.0f ? +1.0f : -1.0f);
          return (v.z < 0.0f) ? ((1.0f - glm::abs(glm::vec2(v.y, v.x))) * ps) : glm::vec2(v.x, v.y);
        };
        const auto encodeDirection = [encodeOctahedral](glm::vec3 v) {
          v = glm::normalize(v);
          glm::vec2 e = encodeOctahedral(v);
          e = e * 0.5f + 0.5f;
          return glm::uintBitsToFloat(glm::packUnorm2x16(e));
        };

        float encodedNormal = encodeDirection(glm::make_vec3(cpuVert.norm));
        float encodedTangent = encodeDirection(glm::make_vec3(cpuVert.tangent));

        allVertices.push_back(Rp::FVertex{
          .field1 = { glm::make_vec3(cpuVert.pos), cpuVert.bitangentSign },
          .field2 = { encodedNormal, encodedTangent, cpuVert.u, cpuVert.v }
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

      // BLAS
      bool isOpaque = s_shaderGen->isMaterialOpaque(mesh->material->sgMat);

      CgpuBlas blas;
      if (!cgpuCreateBlas(s_device, (uint32_t)vertices.size(), vertices.data(),
                          (uint32_t)indices.size(), indices.data(), isOpaque, &blas))
      {
        goto fail_cleanup;
      }

      blases.push_back(blas);

      // FIXME: find a better solution
      uint32_t materialIndex = UINT32_MAX;
      GiShaderCache* shader_cache = params->shaderCache;
      for (uint32_t i = 0; i < shader_cache->materials.size(); i++)
      {
        if (shader_cache->materials[i] == mesh->material)
        {
          materialIndex = i;
          break;
        }
      }
      if (materialIndex == UINT32_MAX)
      {
        goto fail_cleanup;
      }

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

  printf("instance count: %d\n", params->meshInstanceCount);
  printf("creating geom cache..\n");
  fflush(stdout);

  // Build HW ASes and vertex, index buffers.
  CgpuBuffer buffer;
  CgpuTlas tlas;
  std::vector<CgpuBlas> blases;
  std::vector<CgpuBlasInstance> blas_instances;
  std::vector<Rp::FVertex> allVertices;
  std::vector<Rp::Face> allFaces;

  if (!_giBuildGeometryStructures(params, blases, blas_instances, allVertices, allFaces))
    goto cleanup;

  if (!cgpuCreateTlas(s_device, blas_instances.size(), blas_instances.data(), &tlas))
    goto cleanup;

  // Upload vertex & index buffers to single GPU buffer.
  GiGpuBufferView faceBufferView;
  GiGpuBufferView vertexBufferView;
  {
    uint64_t buf_size = 0;
    const uint64_t offset_align = s_deviceProperties.minStorageBufferOffsetAlignment;

    faceBufferView.size = allFaces.size() * sizeof(Rp::Face);
    vertexBufferView.size = allVertices.size() * sizeof(Rp::FVertex);

    faceBufferView.offset = giAlignBuffer(offset_align, faceBufferView.size, &buf_size);
    vertexBufferView.offset = giAlignBuffer(offset_align, vertexBufferView.size, &buf_size);

    printf("total geom buffer size: %.2fMiB\n", buf_size * BYTES_TO_MIB);
    printf("> %.2fMiB faces\n", faceBufferView.size * BYTES_TO_MIB);
    printf("> %.2fMiB vertices\n", vertexBufferView.size * BYTES_TO_MIB);
    fflush(stdout);

    CgpuBufferUsageFlags bufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
    CgpuMemoryPropertyFlags bufferMemProps = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL;

    if (!cgpuCreateBuffer(s_device, bufferUsage, bufferMemProps, buf_size, &buffer))
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
    fprintf(stderr, "error: unsupported AOV - device feature missing\n");
    return nullptr;
  }

  GiScene* scene = params->scene;

  printf("material count: %d\n", params->materialCount);
  printf("creating shader cache..\n");
  fflush(stdout);

  GiShaderCache* cache = nullptr;
  CgpuPipeline pipeline;
  CgpuShader rgenShader;
  std::vector<CgpuShader> missShaders;
  std::vector<CgpuShader> hitShaders;
  std::vector<CgpuImage> images_2d;
  std::vector<CgpuImage> images_3d;
  std::vector<CgpuRtHitGroup> hitGroups;
  std::vector<sg::TextureResource> textureResources;
  uint32_t texCount2d = 0;
  uint32_t texCount3d = 0;
  bool hasPipelineClosestHitShader = false;
  bool hasPipelineAnyHitShader = false;

  // Upload dome light.
  if (scene->domeLight != params->domeLight)
  {
    if (scene->domeLightTexture.handle)
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
        fprintf(stderr, "unable to load dome light texture at %s\n", filePath);
      }
      else
      {
        scene->domeLightTransform = domeLight->transform;
        scene->domeLight = domeLight;
      }
    }
  }

  bool domeLightEnabled = bool(scene->domeLight);

  // Create per-material closest-hit shaders.
  //
  // This is done in multiple phases: first, GLSL is generated from MDL, and
  // texture information is extracted. The information is then used to generated
  // the descriptor sets for the pipeline. Lastly, the GLSL is stiched, #defines
  // are added, and the code is compiled to SPIR-V.
  {
    std::vector<sg::Material*> materials;
    materials.resize(params->materialCount);
    for (int i = 0; i < params->materialCount; i++)
    {
      materials[i] = params->materials[i]->sgMat;
    }

    // 1. Generate GLSL from MDL
    struct HitShaderCompInfo
    {
      sg::ShaderGen::MaterialGlslGenInfo genInfo;
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
      const GiMaterial* mat = params->materials[i];

      HitGroupCompInfo groupInfo;
      {
        sg::ShaderGen::MaterialGlslGenInfo genInfo;
        if (!s_shaderGen->generateMaterialShadingGenInfo(mat->sgMat, genInfo))
        {
          threadWorkFailed = true;
          continue;
        }

        HitShaderCompInfo hitInfo;
        hitInfo.genInfo = genInfo;
        groupInfo.closestHitInfo = hitInfo;
      }
      if (!s_shaderGen->isMaterialOpaque(mat->sgMat))
      {
        sg::ShaderGen::MaterialGlslGenInfo genInfo;
        if (!s_shaderGen->generateMaterialOpacityGenInfo(mat->sgMat, genInfo))
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
    texCount2d += int(domeLightEnabled);

    for (HitGroupCompInfo& groupInfo : hitGroupCompInfos)
    {
      HitShaderCompInfo& closestHitShaderCompInfo = groupInfo.closestHitInfo;
      closestHitShaderCompInfo.texOffset2d = texCount2d;
      closestHitShaderCompInfo.texOffset3d = texCount3d;

      for (const sg::TextureResource& tr : closestHitShaderCompInfo.genInfo.textureResources)
      {
        (tr.is3dImage ? texCount3d : texCount2d)++;
        textureResources.push_back(tr);
      }

      if (groupInfo.anyHitInfo)
      {
        HitShaderCompInfo& anyHitShaderCompInfo = *groupInfo.anyHitInfo;
        anyHitShaderCompInfo.texOffset2d = texCount2d;
        anyHitShaderCompInfo.texOffset3d = texCount3d;

        for (const sg::TextureResource& tr : anyHitShaderCompInfo.genInfo.textureResources)
        {
          (tr.is3dImage ? texCount3d : texCount2d)++;
          textureResources.push_back(tr);
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
      HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

      // Closest hit
      {
        sg::ShaderGen::ClosestHitShaderParams hitParams;
        hitParams.aovId = params->aovId;
        hitParams.baseFileName = "rt_main.chit";
        hitParams.isOpaque = s_shaderGen->isMaterialOpaque(params->materials[i]->sgMat);
        hitParams.nextEventEstimation = params->nextEventEstimation;
        hitParams.shadingGlsl = compInfo.closestHitInfo.genInfo.glslSource;
        hitParams.sphereLightCount = scene->sphereLights.elementCount();
        hitParams.distantLightCount = scene->distantLights.elementCount();
        hitParams.rectLightCount = scene->rectLights.elementCount();
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
        sg::ShaderGen::AnyHitShaderParams hitParams;
        hitParams.aovId = params->aovId;
        hitParams.baseFileName = "rt_main.ahit";
        hitParams.opacityEvalGlsl = compInfo.anyHitInfo->genInfo.glslSource;
        hitParams.sphereLightCount = scene->sphereLights.elementCount();
        hitParams.distantLightCount = scene->distantLights.elementCount();
        hitParams.rectLightCount = scene->rectLights.elementCount();
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

          if (!cgpuCreateShader(s_device, spv.size(), spv.data(), CGPU_SHADER_STAGE_CLOSEST_HIT, &closestHitShader))
          {
            goto cleanup;
          }

          hitShaders.push_back(closestHitShader);
        }

        CgpuShader anyHitShader;
        if (compInfo.anyHitInfo)
        {
          const std::vector<uint8_t>& spv = compInfo.anyHitInfo->spv;

          if (!cgpuCreateShader(s_device, spv.size(), spv.data(), CGPU_SHADER_STAGE_ANY_HIT, &anyHitShader))
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

          if (!cgpuCreateShader(s_device, spv.size(), spv.data(), CGPU_SHADER_STAGE_ANY_HIT, &anyHitShader))
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
    sg::ShaderGen::RaygenShaderParams rgenParams;
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
    rgenParams.shaderClockExts = clockCyclesAov;
    rgenParams.texCount2d = texCount2d;
    rgenParams.texCount3d = texCount3d;

    std::vector<uint8_t> rgenSpirv;
    if (!s_shaderGen->generateRgenSpirv("rt_main.rgen", rgenParams, rgenSpirv))
    {
      goto cleanup;
    }

    if (!cgpuCreateShader(s_device, rgenSpirv.size(), rgenSpirv.data(), CGPU_SHADER_STAGE_RAYGEN, &rgenShader))
    {
      goto cleanup;
    }
  }

  // Create miss shaders.
  {
    sg::ShaderGen::MissShaderParams missParams;
    missParams.domeLightEnabled = domeLightEnabled;
    missParams.domeLightCameraVisibility = params->domeLightCameraVisibility;
    missParams.sphereLightCount = scene->sphereLights.elementCount();
    missParams.distantLightCount = scene->distantLights.elementCount();
    missParams.rectLightCount = scene->rectLights.elementCount();
    missParams.texCount2d = texCount2d;
    missParams.texCount3d = texCount3d;

    // regular miss shader
    {
      std::vector<uint8_t> missSpirv;
      if (!s_shaderGen->generateMissSpirv("rt_main.miss", missParams, missSpirv))
      {
        goto cleanup;
      }

      CgpuShader missShader;
      if (!cgpuCreateShader(s_device, missSpirv.size(), missSpirv.data(), CGPU_SHADER_STAGE_MISS, &missShader))
      {
        goto cleanup;
      }

      missShaders.push_back(missShader);
    }

    // shadow test miss shader
    {
      std::vector<uint8_t> missSpirv;
      if (!s_shaderGen->generateMissSpirv("rt_shadow.miss", missParams, missSpirv))
      {
        goto cleanup;
      }

      CgpuShader missShader;
      if (!cgpuCreateShader(s_device, missSpirv.size(), missSpirv.data(), CGPU_SHADER_STAGE_MISS, &missShader))
      {
        goto cleanup;
      }

      missShaders.push_back(missShader);
    }
  }

  // Upload textures.
  if (textureResources.size() > 0 && !s_texSys->loadTextureResources(textureResources, images_2d, images_3d))
  {
    goto cleanup;
  }
  assert(images_2d.size() == (texCount2d - int(domeLightEnabled)));
  assert(images_3d.size() == texCount3d);

  // Create RT pipeline.
  {
    printf("creating RT pipeline..\n");
    fflush(stdout);

    CgpuRtPipelineDesc pipelineDesc = {
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
  cache->hitShaders = std::move(hitShaders);
  cache->images2d = std::move(images_2d);
  cache->images3d = std::move(images_3d);
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
    s_texSys->destroyUncachedImages(images_2d);
    s_texSys->destroyUncachedImages(images_3d);
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
  s_sampleOffset = 0;
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

  // Init state for goto error handling.
  int result = GI_ERROR;

  if (!scene->sphereLights.commitChanges())
  {
    fprintf(stderr, "%s:%d: light commit failed!\n", __FILE__, __LINE__);
  }
  if (!scene->distantLights.commitChanges())
  {
    fprintf(stderr, "%s:%d: light commit failed!\n", __FILE__, __LINE__);
  }
  if (!scene->rectLights.commitChanges())
  {
    fprintf(stderr, "%s:%d: light commit failed!\n", __FILE__, __LINE__);
  }

  if (!s_stager->flush())
  {
    fprintf(stderr, "%s:%d: stager flush failed!\n", __FILE__, __LINE__);
  }

  CgpuCommandBuffer command_buffer;
  CgpuFence fence;

  // Set up output buffer.
  int compCount = 4;
  int pixelStride = compCount * sizeof(float);
  int pixelCount = params->imageWidth * params->imageHeight;
  uint64_t outputBufferSize = pixelCount * pixelStride;

  bool reallocOutputBuffer = s_outputBufferWidth != params->imageWidth ||
                             s_outputBufferHeight != params->imageHeight;

  if (reallocOutputBuffer)
  {
    printf("recreating output buffer with size %dx%d (%.2fMiB)\n", params->imageWidth,
      params->imageHeight, outputBufferSize * BYTES_TO_MIB);

    _giResizeOutputBuffer(params->imageWidth, params->imageHeight, outputBufferSize);
  }

  // Set up GPU data.
  auto camForward = glm::normalize(glm::make_vec3(params->camera->forward));
  auto camUp = glm::normalize(glm::make_vec3(params->camera->up));

  float lensRadius = 0.0f;
  if (params->camera->fStop > 0.0f)
  {
    lensRadius = params->camera->focalLength / (2.0f * params->camera->fStop);
  }

  Rp::PushConstants pushData = {
    .cameraPosition              = glm::make_vec3(params->camera->position),
    .imageDims                   = ((params->imageHeight << 16) | params->imageWidth),
    .cameraForward               = camForward,
    .focusDistance               = params->camera->focusDistance,
    .cameraUp                    = camUp,
    .cameraVFoV                  = params->camera->vfov,
    .backgroundColor             = glm::make_vec4(params->bgColor),
    .sampleOffset                = s_sampleOffset,
    .lensRadius                  = lensRadius,
    .sampleCount                 = params->spp,
    .maxSampleValue              = params->maxSampleValue,
    .domeLightTransformCol0      = scene->domeLightTransform[0],
    .maxBouncesAndRrBounceOffset = ((params->maxBounces << 16) | params->rrBounceOffset),
    .domeLightTransformCol1      = scene->domeLightTransform[1],
    .rrInvMinTermProb            = params->rrInvMinTermProb,
    .domeLightTransformCol2      = scene->domeLightTransform[2],
    .lightIntensityMultiplier    = params->lightIntensityMultiplier
  };

  std::vector<CgpuBufferBinding> buffers;
  buffers.reserve(16);

  buffers.push_back({ .binding = Rp::BINDING_INDEX_OUT_PIXELS, .buffer = s_outputBuffer });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_FACES, .buffer = geom_cache->buffer,
                      .offset = geom_cache->faceBufferView.offset, .size = geom_cache->faceBufferView.size });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_VERTICES, .buffer = geom_cache->buffer,
                      .offset = geom_cache->vertexBufferView.offset, .size = geom_cache->vertexBufferView.size });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_SPHERE_LIGHTS, .buffer = scene->sphereLights.buffer() });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_DISTANT_LIGHTS, .buffer = scene->distantLights.buffer() });
  buffers.push_back({ .binding = Rp::BINDING_INDEX_RECT_LIGHTS, .buffer = scene->rectLights.buffer() });

  bool domeLightEnabled = bool(scene->domeLight);
  size_t imageCount = shader_cache->images2d.size() + shader_cache->images3d.size() + int(domeLightEnabled);

  std::vector<CgpuImageBinding> images;
  images.reserve(imageCount);

  CgpuSamplerBinding sampler = { .binding = Rp::BINDING_INDEX_SAMPLER, .sampler = s_texSampler };

  if (domeLightEnabled)
  {
    images.push_back({ .binding = Rp::BINDING_INDEX_TEXTURES_2D, .image = scene->domeLightTexture });
  }
  for (uint32_t i = 0; i < shader_cache->images2d.size(); i++)
  {
    images.push_back({ .binding = Rp::BINDING_INDEX_TEXTURES_2D,
                       .image = shader_cache->images2d[i],
                       .index = int(domeLightEnabled) + i });
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
  if (!cgpuCreateCommandBuffer(s_device, &command_buffer))
    goto cleanup;

  if (!cgpuBeginCommandBuffer(command_buffer))
    goto cleanup;

  if (!cgpuCmdTransitionShaderImageLayouts(command_buffer, shader_cache->rgenShader, images.size(), images.data()))
    goto cleanup;

  if (!cgpuCmdUpdateBindings(command_buffer, shader_cache->pipeline, &bindings))
    goto cleanup;

  if (!cgpuCmdBindPipeline(command_buffer, shader_cache->pipeline))
    goto cleanup;

  // Trace rays.
  {
    CgpuShaderStageFlags pushShaderStages = CGPU_SHADER_STAGE_RAYGEN | CGPU_SHADER_STAGE_MISS;
    pushShaderStages |= shader_cache->hasPipelineClosestHitShader ? CGPU_SHADER_STAGE_CLOSEST_HIT : 0;
    pushShaderStages |= shader_cache->hasPipelineAnyHitShader ? CGPU_SHADER_STAGE_ANY_HIT : 0;

    if (!cgpuCmdPushConstants(command_buffer, shader_cache->pipeline, pushShaderStages, sizeof(pushData), &pushData))
      goto cleanup;
  }

  if (!cgpuCmdTraceRays(command_buffer, shader_cache->pipeline, params->imageWidth, params->imageHeight))
    goto cleanup;

  // Copy output buffer to staging buffer.
  {
    CgpuBufferMemoryBarrier barrier = {
      .buffer = s_outputBuffer,
      .srcAccessFlags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE,
      .dstAccessFlags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ
    };

    if (!cgpuCmdPipelineBarrier(command_buffer, 0, nullptr, 1, &barrier, 0, nullptr))
      goto cleanup;
  }

  if (!cgpuCmdCopyBuffer(command_buffer, s_outputBuffer, 0, s_outputStagingBuffer, 0, outputBufferSize))
    goto cleanup;

  {
    CgpuBufferMemoryBarrier barrier = {
      .buffer = s_outputBuffer,
      .srcAccessFlags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE,
      .dstAccessFlags = CGPU_MEMORY_ACCESS_FLAG_HOST_READ,
    };

    if (!cgpuCmdPipelineBarrier(command_buffer, 0, nullptr, 1, &barrier, 0, nullptr))
      goto cleanup;
  }

  // Submit command buffer.
  if (!cgpuEndCommandBuffer(command_buffer))
    goto cleanup;

  if (!cgpuCreateFence(s_device, &fence))
    goto cleanup;

  if (!cgpuResetFence(s_device, fence))
    goto cleanup;

  if (!cgpuSubmitCommandBuffer(s_device, command_buffer, fence))
    goto cleanup;

  // Now is a good time to flush buffered messages (on Windows).
  fflush(stdout);

  if (!cgpuWaitForFence(s_device, fence))
    goto cleanup;

  // Read data from GPU to image.
  uint8_t* mapped_staging_mem;
  if (!cgpuMapBuffer(s_device, s_outputStagingBuffer, (void**) &mapped_staging_mem))
    goto cleanup;

  memcpy(rgbaImg, mapped_staging_mem, outputBufferSize);

  if (!cgpuUnmapBuffer(s_device, s_outputStagingBuffer))
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
      rgbaImg[i + 0] = (float) gi::TURBO_SRGB_FLOATS[val_index][0];
      rgbaImg[i + 1] = (float) gi::TURBO_SRGB_FLOATS[val_index][1];
      rgbaImg[i + 2] = (float) gi::TURBO_SRGB_FLOATS[val_index][2];
      rgbaImg[i + 3] = 255;
    }
  }

  s_sampleOffset += params->spp;

  result = GI_OK;

cleanup:
  cgpuDestroyFence(s_device, fence);
  cgpuDestroyCommandBuffer(s_device, command_buffer);

  return result;
}

GiScene* giCreateScene()
{
  GiScene* scene = new GiScene{
    .sphereLights = GgpuDenseDataStore(s_device, *s_stager, sizeof(Rp::SphereLight), 64),
    .distantLights = GgpuDenseDataStore(s_device, *s_stager, sizeof(Rp::DistantLight), 64),
    .rectLights = GgpuDenseDataStore(s_device, *s_stager, sizeof(Rp::RectLight), 64)
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
  delete scene;
}

GiSphereLight* giCreateSphereLight(GiScene* scene)
{
  auto light = new GiSphereLight;
  light->scene = scene;
  light->gpuHandle = scene->sphereLights.allocate();

  Rp::SphereLight* data = scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
  assert(data);

  data->pos[0] = 0.0f;
  data->pos[1] = 0.0f;
  data->pos[2] = 0.0f;
  data->radius = 0.0f;
  data->baseEmission[0] = 0.0f;
  data->baseEmission[1] = 0.0f;
  data->baseEmission[2] = 0.0f;

  return light;
}

void giDestroySphereLight(GiScene* scene, GiSphereLight* light)
{
  scene->sphereLights.free(light->gpuHandle);
  delete light;
}

void giSetSphereLightPosition(GiSphereLight* light, float* pos)
{
  Rp::SphereLight* data = light->scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
  assert(data);

  data->pos[0] = pos[0];
  data->pos[1] = pos[1];
  data->pos[2] = pos[2];
}

void giSetSphereLightBaseEmission(GiSphereLight* light, float* rgb)
{
  Rp::SphereLight* data = light->scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
  assert(data);

  data->baseEmission[0] = rgb[0];
  data->baseEmission[1] = rgb[1];
  data->baseEmission[2] = rgb[2];
}

void giSetSphereLightRadius(GiSphereLight* light, float radius)
{
  Rp::SphereLight* data = light->scene->sphereLights.write<Rp::SphereLight>(light->gpuHandle);
  assert(data);

  data->radius = radius;
}

GiDistantLight* giCreateDistantLight(GiScene* scene)
{
  auto light = new GiDistantLight;
  light->scene = scene;
  light->gpuHandle = scene->distantLights.allocate();

  Rp::DistantLight* data = scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->direction[0] = 0.0f;
  data->direction[1] = 0.0f;
  data->direction[2] = 0.0f;
  data->angle = 0.0f;
  data->baseEmission[0] = 0.0f;
  data->baseEmission[1] = 0.0f;
  data->baseEmission[2] = 0.0f;

  return light;
}

void giDestroyDistantLight(GiScene* scene, GiDistantLight* light)
{
  scene->distantLights.free(light->gpuHandle);
  delete light;
}

void giSetDistantLightDirection(GiDistantLight* light, float* direction)
{
  Rp::DistantLight* data = light->scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->direction[0] = direction[0];
  data->direction[1] = direction[1];
  data->direction[2] = direction[2];
}

void giSetDistantLightBaseEmission(GiDistantLight* light, float* rgb)
{
  Rp::DistantLight* data = light->scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->baseEmission[0] = rgb[0];
  data->baseEmission[1] = rgb[1];
  data->baseEmission[2] = rgb[2];
}

void giSetDistantLightAngle(GiDistantLight* light, float angle)
{
  Rp::DistantLight* data = light->scene->distantLights.write<Rp::DistantLight>(light->gpuHandle);
  assert(data);

  data->angle = angle;
}

GiRectLight* giCreateRectLight(GiScene* scene)
{
  auto light = new GiRectLight;
  light->scene = scene;
  light->gpuHandle = scene->rectLights.allocate();

  Rp::RectLight* data = scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->origin[0] = 0.0f;
  data->origin[1] = 0.0f;
  data->origin[2] = 0.0f;
  data->width = 0.0f;
  data->direction[0] = 0.0f;
  data->direction[1] = 0.0f;
  data->direction[2] = 0.0f;
  data->height = 0.0f;
  data->baseEmission[0] = 0.0f;
  data->baseEmission[1] = 0.0f;
  data->baseEmission[2] = 0.0f;

  return light;
}

void giDestroyRectLight(GiScene* scene, GiRectLight* light)
{
  scene->rectLights.free(light->gpuHandle);
  delete light;
}

void giSetRectLightOrigin(GiRectLight* light, float* origin)
{
  Rp::RectLight* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->origin[0] = origin[0];
  data->origin[1] = origin[1];
  data->origin[2] = origin[2];
}

void giSetRectLightDirection(GiRectLight* light, float* direction)
{
  Rp::RectLight* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->direction[0] = direction[0];
  data->direction[1] = direction[1];
  data->direction[2] = direction[2];
}

void giSetRectLightBaseEmission(GiRectLight* light, float* rgb)
{
  Rp::RectLight* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->baseEmission[0] = rgb[0];
  data->baseEmission[1] = rgb[1];
  data->baseEmission[2] = rgb[2];
}

void giSetRectLightDimensions(GiRectLight* light, float width, float height)
{
  Rp::RectLight* data = light->scene->rectLights.write<Rp::RectLight>(light->gpuHandle);
  assert(data);

  data->width = width;
  data->height = height;
}

GiDomeLight* giCreateDomeLight(GiScene* scene, const char* filePath)
{
  GiDomeLight* light = new GiDomeLight;
  light->textureFilePath = filePath;
  return light;
}

void giDestroyDomeLight(GiScene* scene, GiDomeLight* light)
{
  delete light;
}

void giSetDomeLightTransform(GiDomeLight* light, float* transformPtr)
{
  auto transform = glm::inverse(glm::make_mat3(transformPtr));
  memcpy(&light->transform, glm::value_ptr(transform), sizeof(transform));
}
