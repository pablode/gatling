//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include "MaterialManager.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <optional>
#include <unordered_map>

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/gb/ThreadPool.h>
#include <gtl/ggpu/DeleteQueue.h>
#include <gtl/ggpu/Stager.h>
#include <gtl/mc/Material.h>

namespace gtl
{
  namespace
  {
    void _DestroyMaterialGpuData(CgpuContext* ctx,
                                 OffsetAllocator::Allocator& texAllocator,
                                 GiMaterialGpuData& gpuData)
    {
      if (gpuData.texOffsetAllocation.offset != OffsetAllocator::Allocation::NO_SPACE)
      {
        texAllocator.free(gpuData.texOffsetAllocation);
        gpuData.texOffsetAllocation = {};
      }

      if (gpuData.closestHit.handle)
      {
        cgpuDestroyShader(ctx, gpuData.closestHit);
        gpuData.closestHit = {};
      }

      for (CgpuShader shader : gpuData.anyHits)
      {
        cgpuDestroyShader(ctx, shader);
      }
      gpuData.anyHits.clear();
    }

    const McMtlxTextureNetworkParameter* _FindTextureParameter(const GiBaseMaterial& material,
                                                               const std::string& mdlParameterName)
    {
      for (const McMtlxNetworkParameter& parameter : material.networkParameters)
      {
        const McMtlxTextureNetworkParameter* textureParameter = std::get_if<McMtlxTextureNetworkParameter>(&parameter);
        if (textureParameter && textureParameter->mdlParameterName == mdlParameterName)
        {
          return textureParameter;
        }
      }

      return nullptr;
    }

    rp::TextureInfo _MakeTextureInfo(const GiTextureResource& resource, uint32_t imageIndex)
    {
      assert(imageIndex <= rp::TEXTURE_INFO_IMAGE_INDEX_MASK);
      assert(resource.uvtileMinU >= INT8_MIN && resource.uvtileMinU <= INT8_MAX);
      assert(resource.uvtileMinV >= INT8_MIN && resource.uvtileMinV <= INT8_MAX);
      assert(resource.uvtileWidth <= rp::TEXTURE_INFO_UVTILE_SIZE_MASK);
      assert(resource.uvtileHeight <= rp::TEXTURE_INFO_UVTILE_SIZE_MASK);

      rp::TextureInfo texInfo = {};
      texInfo.imageIndexAndUvtileMin =
        imageIndex |
        (uint32_t(resource.uvtileMinU & 0xFF) << rp::TEXTURE_INFO_UVTILE_MIN_U_SHIFT) |
        (uint32_t(resource.uvtileMinV & 0xFF) << rp::TEXTURE_INFO_UVTILE_MIN_V_SHIFT);
      texInfo.uvtileWidthAndHeight =
        resource.uvtileWidth | (resource.uvtileHeight << rp::TEXTURE_INFO_UVTILE_HEIGHT_SHIFT);
      return texInfo;
    }

    // Materials can only share GPU data if their network topology and all properties that
    // affect generated shader code are equal. Most of the latter are derived from the compiled
    // material, which contains the per-instance parameter values. The parameter names are part
    // of the key because they determine the scene data indices baked into the shader. They are
    // identical only if the parameter traversal order is identical.
    uint64_t _MakeMaterialGroupKey(const GiBaseMaterial& material)
    {
      const McMaterial& mcMat = *material.mcMat;

      GbHash hash = GbHashAppend(GbHash{}, material.topoHash.val);
      hash = GbHashAppend(hash, uint32_t(mcMat.opacityClass));
      hash = GbHashAppend(hash, mcMat.hasBackfaceBsdf);
      hash = GbHashAppend(hash, mcMat.hasBackfaceEdf);
      hash = GbHashAppend(hash, mcMat.hasVolumeAbsorptionCoeff);
      hash = GbHashAppend(hash, mcMat.hasVolumeScatteringCoeff);
      hash = GbHashAppend(hash, mcMat.isAnimated);
      hash = GbHashAppend(hash, mcMat.isEmissive);
      hash = GbHashAppend(hash, mcMat.isThinWalled);
      hash = GbHashAppend(hash, mcMat.directionalBias);
      hash = GbHashAppend(hash, mcMat.requiresSceneTransforms);

      for (const McMtlxNetworkParameter& parameter : material.networkParameters)
      {
        hash = GbHashAppend(hash, McMtlxGetNetworkParameterName(parameter));
        hash = GbHashAppend(hash, std::holds_alternative<McMtlxTextureNetworkParameter>(parameter));
      }

      return hash.val;
    }
  }

  GiMaterialManager::GiMaterialManager(CgpuContext* ctx,
                                       GiGlslShaderGen& shaderGen,
                                       GiTextureManager& texSys,
                                       GgpuStager& stager,
                                       GgpuDeleteQueue& deleteQueue,
                                       GbThreadPool& threadPool,
                                       OffsetAllocator::Allocator& texAllocator)
    : m_ctx(ctx)
    , m_shaderGen(shaderGen)
    , m_texSys(texSys)
    , m_stager(stager)
    , m_deleteQueue(deleteQueue)
    , m_threadPool(threadPool)
    , m_texAllocator(texAllocator)
  {
  }

  void GiMaterialManager::destroy()
  {
    std::vector<CgpuBuffer> parameterBuffers;
    std::vector<GiMaterialGpuDataPtr> gpuDatas;
    std::vector<OffsetAllocator::Allocation> textureAllocations;
    std::vector<std::vector<GiImagePtr>> textureParameterImages;

    {
      std::lock_guard guard(m_mutex);

      parameterBuffers.swap(m_pendingParameterBuffers);
      gpuDatas.swap(m_pendingGpuData);
      textureAllocations.swap(m_pendingTextureAllocations);
      textureParameterImages.swap(m_pendingTextureParameterImages);

      for (GiBaseMaterial* material : m_materials)
      {
        if (material->parameterBuffer.handle)
        {
          parameterBuffers.push_back(material->parameterBuffer);
          material->parameterBuffer = {};
        }

        if (material->textureParameterAllocation.offset != OffsetAllocator::Allocation::NO_SPACE)
        {
          textureAllocations.push_back(material->textureParameterAllocation);
          material->textureParameterAllocation = {};
        }

        if (!material->textureParameterImages.empty())
        {
          textureParameterImages.push_back(std::move(material->textureParameterImages));
        }

        if (material->gpuData)
        {
          gpuDatas.push_back(std::move(material->gpuData));
        }
      }

      m_materials.clear();
    }

    for (CgpuBuffer buffer : parameterBuffers)
    {
      cgpuDestroyBuffer(m_ctx, buffer);
    }

    for (OffsetAllocator::Allocation allocation : textureAllocations)
    {
      m_texAllocator.free(allocation);
    }

    // Releasing the parameter images enqueues their GPU destruction.
    textureParameterImages.clear();

    // Custom deleters run here.
    gpuDatas.clear();
  }

  void GiMaterialManager::addMaterial(GiBaseMaterial* material)
  {
    std::lock_guard guard(m_mutex);
    m_materials.insert(material);
  }

  void GiMaterialManager::removeMaterial(GiBaseMaterial* material)
  {
    std::lock_guard guard(m_mutex);

    if (material->parameterBuffer.handle)
    {
      // cgpu's VMA allocator requires external synchronization. Destruction is deferred
      // to the render thread, which owns all other GPU resource lifetime operations.
      m_pendingParameterBuffers.push_back(material->parameterBuffer);
      material->parameterBuffer = {};
    }

    if (material->gpuData)
    {
      if (material->gpuData.use_count() == 1)
      {
        m_pendingGpuData.push_back(std::move(material->gpuData));
      }
      else
      {
        material->gpuData.reset();
      }
    }

    if (material->textureParameterAllocation.offset != OffsetAllocator::Allocation::NO_SPACE)
    {
      m_pendingTextureAllocations.push_back(material->textureParameterAllocation);
      material->textureParameterAllocation = {};
    }

    if (!material->textureParameterImages.empty())
    {
      // Image destruction enqueues GPU work, so it is deferred to the render thread.
      m_pendingTextureParameterImages.push_back(std::move(material->textureParameterImages));
    }

    m_materials.erase(material);
  }

  std::vector<GiBaseMaterial*> GiMaterialManager::materials()
  {
    std::lock_guard guard(m_mutex);
    return std::vector<GiBaseMaterial*>(m_materials.begin(), m_materials.end());
  }

  void GiMaterialManager::housekeep()
  {
    std::vector<CgpuBuffer> parameterBuffers;
    std::vector<GiMaterialGpuDataPtr> gpuDatas;
    std::vector<OffsetAllocator::Allocation> textureAllocations;
    std::vector<std::vector<GiImagePtr>> textureParameterImages;

    {
      std::lock_guard guard(m_mutex);
      parameterBuffers.swap(m_pendingParameterBuffers);
      gpuDatas.swap(m_pendingGpuData);
      textureAllocations.swap(m_pendingTextureAllocations);
      textureParameterImages.swap(m_pendingTextureParameterImages);
    }

    for (CgpuBuffer buffer : parameterBuffers)
    {
      m_deleteQueue.pushBack(buffer);
    }

    for (OffsetAllocator::Allocation allocation : textureAllocations)
    {
      m_texAllocator.free(allocation);
    }

    // Releasing the parameter images enqueues their GPU destruction.
    textureParameterImages.clear();

    // Custom deleters run here, on the render thread.
    gpuDatas.clear();
  }

  GiMaterialGpuDataPtr GiMaterialManager::makeGpuDataPtr(GiMaterialGpuData&& gpuData)
  {
    // Capture the raw pointers instead of 'this' so that materials which outlive the
    // manager can still be destroyed safely.
    CgpuContext* ctx = m_ctx;
    OffsetAllocator::Allocator* texAllocator = &m_texAllocator;

    return GiMaterialGpuDataPtr(new GiMaterialGpuData(std::move(gpuData)),
                                [ctx, texAllocator](GiMaterialGpuData* data)
                                {
                                  _DestroyMaterialGpuData(ctx, *texAllocator, *data);
                                  delete data;
                                });
  }

  bool GiMaterialManager::ensureParameterBuffers(const std::vector<GiBaseMaterial*>& materials)
  {
    std::vector<GiBaseMaterial*> createdMaterials;

    const auto cleanup = [&]()
    {
      for (GiBaseMaterial* material : createdMaterials)
      {
        cgpuDestroyBuffer(m_ctx, material->parameterBuffer);
        material->parameterBuffer = {};
      }
    };

    for (GiBaseMaterial* material : materials)
    {
      if (material->parameterBuffer.handle)
      {
        continue;
      }

      uint64_t bufferSize = std::max<size_t>(material->parameterData.size(), rp::SCENE_DATA_ALIGNMENT);
      std::vector<uint8_t> data = material->parameterData;
      data.resize(bufferSize, 0);

      std::string debugName = GB_FMT("MaterialParams_{}", material->name);

      CgpuBuffer buffer;
      if (!cgpuCreateBuffer(m_ctx, {
                              .usage = CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::TransferDst,
                              .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                              .size = bufferSize,
                              .debugName = debugName.c_str()
                            }, &buffer))
      {
        GB_ERROR("failed to allocate material parameter buffer");
        cleanup();
        return false;
      }

      if (!m_stager.stageToBuffer(data.data(), bufferSize, buffer))
      {
        GB_ERROR("failed to stage material parameter buffer");
        cgpuDestroyBuffer(m_ctx, buffer);
        cleanup();
        return false;
      }

      material->parameterBuffer = buffer;
      createdMaterials.push_back(material);
    }

    if (!createdMaterials.empty() && !m_stager.flush())
    {
      cleanup();
      return false;
    }

    return true;
  }

  bool GiMaterialManager::compileMaterials(const std::vector<GiBaseMaterial*>& materials,
                                           const GiGlslShaderGen::CommonShaderParams& commonParams,
                                           uint32_t maxRayPayloadSize,
                                           uint32_t maxRayHitAttributeSize,
                                           bool nextEventEstimation,
                                           bool needsAuxOutput,
                                           bool disableDeduplication,
                                           GiMaterialCompileResult& result)
  {
    struct ShaderCompileInfo
    {
      std::vector<uint8_t> spv;
      std::vector<uint8_t> shadowSpv;
    };
    struct MaterialCompileInfo
    {
      GiBaseMaterial* material = nullptr;
      GiGlslShaderGen::MaterialGenInfo genInfo;

      ShaderCompileInfo closestHitInfo;
      std::optional<ShaderCompileInfo> anyHitInfo;
    };

    result = {};

    // Materials are grouped by their topological hash and all other properties that affect the
    // generated shaders, so that identical materials share compiled GPU data (shaders, textures
    // and texture infos).
    std::unordered_map<uint64_t, GiBaseMaterial*> baseMaterialByKey;
    std::vector<GiBaseMaterial*> baseMaterials(materials.size());

    // Prefer already compiled materials as group base materials.
    if (!disableDeduplication)
    {
      for (GiBaseMaterial* material : materials)
      {
        if (material->topoHash.val != 0 && material->gpuData)
        {
          baseMaterialByKey.try_emplace(_MakeMaterialGroupKey(*material), material);
        }
      }
    }

    size_t deduplicatedMaterialCount = 0;
    for (size_t i = 0; i < materials.size(); i++)
    {
      GiBaseMaterial* material = materials[i];

      if (disableDeduplication || material->topoHash.val == 0)
      {
        // MDL materials and materials without a valid hash are not deduplicated.
        baseMaterials[i] = material;
        continue;
      }

      auto [it, inserted] = baseMaterialByKey.try_emplace(_MakeMaterialGroupKey(*material), material);
      baseMaterials[i] = it->second;

      if (it->second != material)
      {
        deduplicatedMaterialCount++;
        GB_DEBUG("material {} deduplicated to {}", material->name, it->second->name);
      }
    }

    GB_LOG("unique materials: {} (total: {})", materials.size() - deduplicatedMaterialCount, materials.size());

    bool success = false;
    std::vector<GiMaterialGpuData> newMaterialGpuDatas;
    std::unordered_map<const GiBaseMaterial*, const GiMaterialGpuData*> resolvedGpuData;
    std::vector<McTextureDescription> textureDescriptions;
    std::vector<GiImageBinding> imageBindings;
    std::vector<std::pair<uint32_t, rp::TextureInfo>> textureInfoEntries;
    std::vector<MaterialCompileInfo> materialCompileInfos;
    std::vector<const GiBaseMaterial*> cachedMaterials;
    std::vector<GiBaseMaterial*> materialsWithTextureParameters;
    uint32_t maxTextureIndex = 0;

    do
    {
      // 1. Generate GLSL from MDL in parallel.
      {
        std::mutex cachedMaterialsMutex;
        std::mutex materialCompileInfoMutex;
        materialCompileInfos.reserve(materials.size());

        std::atomic_bool threadWorkFailed = false;
        m_threadPool.parallelFor((uint32_t) materials.size(), [&](uint32_t i) {
          GiBaseMaterial* material = materials[i];

          if (baseMaterials[i] != material)
          {
            return; // reuses the group's base material GPU data
          }

          if (material->gpuData)
          {
            std::lock_guard guard(cachedMaterialsMutex);
            cachedMaterials.push_back(material);
            return;
          }

          GiGlslShaderGen::MaterialGenInfo genInfo;
          if (!m_shaderGen.generateMaterialInfo(*material->mcMat, genInfo, needsAuxOutput))
          {
            GB_ERROR("failed to generate code for material {}", material->name);
            threadWorkFailed = true;
            return;
          }

          MaterialCompileInfo groupInfo;
          groupInfo.material = material;
          groupInfo.genInfo = std::move(genInfo);

          if (material->mcMat->opacityClass == McOpacityClassification::Cutout)
          {
            groupInfo.anyHitInfo = ShaderCompileInfo{};
          }

          std::lock_guard guard(materialCompileInfoMutex);
          materialCompileInfos.push_back(std::move(groupInfo));
        });
        if (threadWorkFailed)
        {
          goto fail;
        }
      }

      assert(materialCompileInfos.size() + cachedMaterials.size() == materials.size() - deduplicatedMaterialCount);
      GB_LOG("compiling {} material variants ({} cached)", materialCompileInfos.size(), cachedMaterials.size());

      // 2. Collect textures to load.
      for (MaterialCompileInfo& groupInfo : materialCompileInfos)
      {
        const GiGlslShaderGen::MaterialGenInfo& genInfo = groupInfo.genInfo;
        for (const McTextureDescription& tr : genInfo.textureDescriptions)
        {
          textureDescriptions.push_back(tr);
        }
      }

      // 3. Upload textures and assign images to new material GPU data.
      {
        std::vector<GiTextureResource> textureResources;
        if (textureDescriptions.size() > 0 && !m_texSys.loadTextureDescriptions(textureDescriptions, textureResources))
        {
          goto fail;
        }

        uint32_t resourceCounter = 0;
        newMaterialGpuDatas.reserve(materialCompileInfos.size());
        for (MaterialCompileInfo& groupInfo : materialCompileInfos)
        {
          const GiGlslShaderGen::MaterialGenInfo& genInfo = groupInfo.genInfo;
          auto resourceCount = uint32_t(genInfo.textureDescriptions.size());
          uint32_t resourceStart = resourceCounter;
          resourceCounter += resourceCount;

          uint32_t imageCount = 0;
          for (uint32_t r = 0; r < resourceCount; r++)
          {
            const GiTextureResource& resource = textureResources[resourceStart + r];
            imageCount += resource.uvtileWidth > 0 ? resource.uvtileWidth * resource.uvtileHeight : 1;
          }

          GiMaterialGpuData gpuData;
          gpuData.stringConstants = genInfo.stringConstants;
          gpuData.textureInfos.resize(resourceCount);
          gpuData.textureParameterBindings = genInfo.textureParameterBindings;

          if (imageCount > 0)
          {
            gpuData.texOffsetAllocation = m_texAllocator.allocate(imageCount);

            GB_EXPECT(gpuData.texOffsetAllocation.offset != OffsetAllocator::Allocation::NO_SPACE, "max number of textures exceeded");
          }

          uint32_t imageOffset = 0;
          for (uint32_t r = 0; r < resourceCount; r++)
          {
            const GiTextureResource& resource = textureResources[resourceStart + r];

            uint32_t imageIndex = gpuData.texOffsetAllocation.offset + imageOffset;

            if (resource.uvtileWidth > 0)
            {
              gpuData.hasUdimTextures = true;
            }

            gpuData.textureInfos[r] = _MakeTextureInfo(resource, imageIndex);

            uint32_t imageSlotCount = resource.uvtileWidth > 0 ? resource.uvtileWidth * resource.uvtileHeight : 1;
            assert(resource.images.size() == imageSlotCount);

            for (uint32_t s = 0; s < imageSlotCount; s++)
            {
              gpuData.images.push_back(resource.images[s]);
            }

            imageOffset += imageSlotCount;
          }

          newMaterialGpuDatas.push_back(std::move(gpuData));
        }

        // Record the GPU data per base material for parameter resolution.
        for (size_t i = 0; i < materialCompileInfos.size(); i++)
        {
          resolvedGpuData[materialCompileInfos[i].material] = &newMaterialGpuDatas[i];
        }
        for (const GiBaseMaterial* material : cachedMaterials)
        {
          resolvedGpuData[material] = material->gpuData.get();
        }

        // Deduplicated materials reuse the GPU data of their group's base material.
        for (size_t i = 0; i < materials.size(); i++)
        {
          GiBaseMaterial* material = materials[i];

          if (resolvedGpuData.count(material) == 0)
          {
            auto it = resolvedGpuData.find(baseMaterials[i]);
            assert(it != resolvedGpuData.end());
            resolvedGpuData[material] = it->second;
          }
        }
      }

      // 3.5 Resolve texture parameters, which vary between materials sharing a shader.
      for (size_t i = 0; i < materials.size(); i++)
      {
        GiBaseMaterial* material = materials[i];
        GiBaseMaterial* baseMaterial = baseMaterials[i];

        const GiMaterialGpuData& baseMaterialGpuData = *resolvedGpuData.at(baseMaterial);
        const std::vector<McTextureParameterBinding>& textureParameterBindings = baseMaterialGpuData.textureParameterBindings;

        if (textureParameterBindings.empty())
        {
          continue; // nothing to resolve for this material
        }

        materialsWithTextureParameters.push_back(material);

        // All materials sharing a shader address their texture parameters through the
        // same slots, which are based on the shader's parameter count. The slots are only
        // used by the shader, so they must be laid out identically for all materials.
        size_t slotBase = baseMaterial->networkParameters.size();
        size_t slotCount = baseMaterialGpuData.textureInfos.size();
        size_t requiredSize = std::max<size_t>(slotBase + slotCount, 1) * rp::SCENE_DATA_ALIGNMENT;
        if (material->parameterData.size() < requiredSize)
        {
          material->parameterData.resize(requiredSize, 0);
        }
        for (size_t s = slotBase; s < slotBase + slotCount; s++)
        {
          uint32_t invalidIndex = 0xFFFFFFFFu;
          memcpy(material->parameterData.data() + s * rp::SCENE_DATA_ALIGNMENT, &invalidIndex, sizeof(invalidIndex));
        }

        // Free the previous allocation, which belonged to a stale shader cache.
        if (material->textureParameterAllocation.offset != OffsetAllocator::Allocation::NO_SPACE)
        {
          m_texAllocator.free(material->textureParameterAllocation);
          material->textureParameterAllocation = {};
          material->textureParameterImages.clear();
        }

        std::vector<McTextureDescription> parameterDescs;
        std::vector<uint32_t> parameterTextureSlots;
        for (const McTextureParameterBinding& binding : textureParameterBindings)
        {
          if (binding.textureIndex < 1 || binding.textureIndex > slotCount)
          {
            GB_ERROR("texture index {} out of range for material {} ({} slots)",
                     binding.textureIndex, material->name, slotCount);
            continue;
          }

          const McMtlxTextureNetworkParameter* textureParameter =
            _FindTextureParameter(*material, binding.parameterName);
          if (!textureParameter)
          {
            GB_ERROR("material {} has no texture parameter {}", material->name, binding.parameterName);
            continue;
          }

          if (textureParameter->filePath.empty())
          {
            continue;
          }

          McTextureDescription desc = {};
          desc.filePath = textureParameter->filePath;

          parameterDescs.push_back(std::move(desc));
          parameterTextureSlots.push_back(binding.textureIndex);
        }

        if (parameterDescs.empty())
        {
          continue;
        }

        std::vector<GiTextureResource> resources;
        if (!m_texSys.loadTextureDescriptions(parameterDescs, resources))
        {
          goto fail;
        }

        uint32_t imageCount = 0;
        for (const GiTextureResource& resource : resources)
        {
          imageCount += resource.uvtileWidth > 0 ? resource.uvtileWidth * resource.uvtileHeight : 1;
        }

        material->textureParameterAllocation = m_texAllocator.allocate(imageCount);
        GB_EXPECT(material->textureParameterAllocation.offset != OffsetAllocator::Allocation::NO_SPACE, "max number of textures exceeded");

        uint32_t imageOffset = 0;
        for (size_t r = 0; r < resources.size(); r++)
        {
          const GiTextureResource& resource = resources[r];
          uint32_t textureInfoIndex = material->textureParameterAllocation.offset + imageOffset;

          rp::TextureInfo texInfo = _MakeTextureInfo(resource, textureInfoIndex);
          textureInfoEntries.push_back({ textureInfoIndex, texInfo });

          uint32_t imageSlotCount = resource.uvtileWidth > 0 ? resource.uvtileWidth * resource.uvtileHeight : 1;
          assert(resource.images.size() == imageSlotCount);

          for (uint32_t s = 0; s < imageSlotCount; s++)
          {
            uint32_t bindingIndex = material->textureParameterAllocation.offset + imageOffset;
            imageBindings.push_back({ .image = resource.images[s], .index = bindingIndex });
            material->textureParameterImages.push_back(resource.images[s]);
            maxTextureIndex = std::max(maxTextureIndex, bindingIndex);
            imageOffset++;
          }

          size_t parameterSlot = slotBase + parameterTextureSlots[r] - 1;
          memcpy(material->parameterData.data() + parameterSlot * rp::SCENE_DATA_ALIGNMENT, &textureInfoIndex, sizeof(textureInfoIndex));
        }
      }

      // Parameter buffer contents change whenever texture LUT indices are reassigned.
      for (GiBaseMaterial* material : materialsWithTextureParameters)
      {
        if (material->parameterBuffer.handle)
        {
          m_deleteQueue.pushBack(material->parameterBuffer);
          material->parameterBuffer = {};
        }
      }

      if (!ensureParameterBuffers(materials))
      {
        goto fail;
      }

      // 4. Generate final hit shader GLSL sources.
      {
        std::atomic_bool threadWorkFailed = false;
        m_threadPool.parallelFor((uint32_t) materialCompileInfos.size(), [&](uint32_t i) {
          MaterialCompileInfo& compInfo = materialCompileInfos[i];

          GiBaseMaterial* material = compInfo.material; // group base material
          const McMaterial* mcMat = material->mcMat;

          uint32_t texOffset = newMaterialGpuDatas[i].texOffsetAllocation.offset;
          bool hasTextureParameters = !newMaterialGpuDatas[i].textureParameterBindings.empty();
          bool useTextureIndexLut = newMaterialGpuDatas[i].hasUdimTextures || hasTextureParameters;
          uint32_t textureParameterSlotBase = uint32_t(material->networkParameters.size());

          // Closest hit
          {
            bool evaluateOpacity = (mcMat->opacityClass == McOpacityClassification::NonBinary);

            GiGlslShaderGen::ClosestHitShaderParams hitParams = {
              .baseFileName = "rp_main.chit",
              .commonParams = commonParams,
              .directionalBias = mcMat->directionalBias,
              .enableSceneTransforms = mcMat->requiresSceneTransforms,
              .builtinSceneDataIndices = compInfo.genInfo.builtinSceneDataIndices,
              .hasBackfaceBsdf = mcMat->hasBackfaceBsdf,
              .hasBackfaceEdf = mcMat->hasBackfaceEdf,
              .evaluateOpacity = evaluateOpacity,
              .hasVolumeAbsorptionCoeff = mcMat->hasVolumeAbsorptionCoeff,
              .hasVolumeScatteringCoeff = mcMat->hasVolumeScatteringCoeff,
              .isAnimated = mcMat->isAnimated,
              .isEmissive = mcMat->isEmissive,
              .isThinWalled = mcMat->isThinWalled,
              .nextEventEstimation = nextEventEstimation,
              .maxSceneDataId = compInfo.genInfo.maxSceneDataId,
              .dfGlsl = compInfo.genInfo.glslSource,
              .textureIndexOffset = texOffset,
              .useTextureIndexLut = useTextureIndexLut,
              .hasTextureParameters = hasTextureParameters,
              .textureParameterSlotBase = textureParameterSlotBase
            };

            if (!m_shaderGen.generateClosestHitSpirv(hitParams, compInfo.closestHitInfo.spv))
            {
              threadWorkFailed = true;
              return;
            }
          }

          assert(!compInfo.anyHitInfo || mcMat->opacityClass == McOpacityClassification::Cutout);

          // Any hit
          if (compInfo.anyHitInfo)
          {
            GiGlslShaderGen::AnyHitShaderParams hitParams = {
              .baseFileName = "rp_main.ahit",
              .commonParams = commonParams,
              .enableSceneTransforms = mcMat->requiresSceneTransforms,
              .builtinSceneDataIndices = compInfo.genInfo.builtinSceneDataIndices,
              .dfGlsl = compInfo.genInfo.glslSource,
              .maxSceneDataId = compInfo.genInfo.maxSceneDataId,
              .textureIndexOffset = texOffset,
              .isAnimated = mcMat->isAnimated,
              .useTextureIndexLut = useTextureIndexLut,
              .hasTextureParameters = hasTextureParameters,
              .textureParameterSlotBase = textureParameterSlotBase
            };

            hitParams.shadowTest = false;
            if (!m_shaderGen.generateAnyHitSpirv(hitParams, compInfo.anyHitInfo->spv))
            {
              threadWorkFailed = true;
              return;
            }

            hitParams.shadowTest = true;
            if (!m_shaderGen.generateAnyHitSpirv(hitParams, compInfo.anyHitInfo->shadowSpv))
            {
              threadWorkFailed = true;
              return;
            }
          }
        });
        if (threadWorkFailed)
        {
          goto fail;
        }
      }

      // 5. Compile & assign back the shaders.
      {
        auto compileInfoCount = (uint32_t) materialCompileInfos.size();

        std::vector<CgpuShaderCreateInfo> createInfos;
        createInfos.reserve(compileInfoCount * 2);

        for (uint32_t i = 0; i < compileInfoCount; i++)
        {
          const MaterialCompileInfo& compInfo = materialCompileInfos[i];

          // regular hit shader
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

          // shadow hit shader
          if (compInfo.anyHitInfo)
          {
            const std::vector<uint8_t>& aSpv = compInfo.anyHitInfo->shadowSpv;
            createInfos.push_back({ .size = aSpv.size(), .source = aSpv.data(), .stageFlags = CgpuShaderStage::AnyHit,
                                    .maxRayPayloadSize = maxRayPayloadSize, .maxRayHitAttributeSize = maxRayHitAttributeSize, });
          }
        }

        std::vector<CgpuShader> shaders(createInfos.size());
        if (!cgpuCreateShadersParallel(m_ctx, (uint32_t) createInfos.size(), createInfos.data(), shaders.data()))
        {
          goto fail;
        }

        for (uint32_t i = 0, hitShaderCounter = 0; i < compileInfoCount; i++)
        {
          const MaterialCompileInfo& compInfo = materialCompileInfos[i];
          GiMaterialGpuData& gpuData = newMaterialGpuDatas[i];

          // regular hit shader
          {
            gpuData.closestHit = shaders[hitShaderCounter++];

            if (compInfo.anyHitInfo)
            {
              gpuData.anyHits.push_back(shaders[hitShaderCounter++]);
            }
          }

          // shadow hit shader
          if (compInfo.anyHitInfo)
          {
            gpuData.anyHits.push_back(shaders[hitShaderCounter++]);
          }
        }
      }

      // Collect image bindings and texture resource infos.
      {
        std::unordered_set<const GiMaterialGpuData*> processedGpuDatas;

        for (size_t i = 0; i < materials.size(); i++)
        {
          const GiBaseMaterial* material = materials[i];
          const GiMaterialGpuData& gpuData = *resolvedGpuData[material];

          if (!processedGpuDatas.insert(&gpuData).second)
          {
            continue;
          }

          uint32_t texOffset = gpuData.texOffsetAllocation.offset;

          uint32_t texCount = 0;
          for (const GiImagePtr& img : gpuData.images)
          {
            uint32_t index = texOffset + texCount;
            imageBindings.push_back({ .image = img, .index = index });
            if (index > maxTextureIndex)
            {
              maxTextureIndex = index;
            }
            texCount++;
          }

          for (uint32_t r = 0; r < gpuData.textureInfos.size(); r++)
          {
            uint32_t textureIndex = texOffset + r;
            textureInfoEntries.push_back({ textureIndex, gpuData.textureInfos[r] });
            maxTextureIndex = std::max(maxTextureIndex, textureIndex);
          }
        }
      }

      // Assign (possibly shared) GPU data to materials. The custom deleter ensures that
      // shared resources are only released when the last material referencing them dies.
      {
        std::vector<GiMaterialGpuDataPtr> newGpuData(materialCompileInfos.size());
        for (size_t i = 0; i < materialCompileInfos.size(); i++)
        {
          MaterialCompileInfo& compInfo = materialCompileInfos[i];
          newGpuData[i] = makeGpuDataPtr(std::move(newMaterialGpuDatas[i]));
          compInfo.material->gpuData = newGpuData[i];
        }

        for (size_t i = 0; i < materials.size(); i++)
        {
          GiBaseMaterial* material = materials[i];
          GiBaseMaterial* baseMaterial = baseMaterials[i];

          if (baseMaterial != material)
          {
            material->gpuData = baseMaterial->gpuData;
          }
        }
      }

      result.imageBindings = std::move(imageBindings);
      result.textureInfoEntries = std::move(textureInfoEntries);
      result.maxTextureIndex = maxTextureIndex;
      success = true;
    } while (false);

  fail:
    if (!success)
    {
      for (GiMaterialGpuData& gpuData : newMaterialGpuDatas)
      {
        _DestroyMaterialGpuData(m_ctx, m_texAllocator, gpuData);
      }

      for (GiBaseMaterial* material : materialsWithTextureParameters)
      {
        if (material->textureParameterAllocation.offset != OffsetAllocator::Allocation::NO_SPACE)
        {
          m_texAllocator.free(material->textureParameterAllocation);
          material->textureParameterAllocation = {};
          material->textureParameterImages.clear();
        }
      }
    }

    return success;
  }
}
