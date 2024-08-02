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

#include "ShaderCacheFactory.h"

#include "GlslShaderGen.h"
#include "TextureManager.h"

#include <gtl/gb/Log.h>
#include <gtl/mc/Material.h>
#include <gtl/mc/Backend.h>

namespace
{
  using namespace gtl;

  bool _MakeMaterialGenInfo(const McGlslGenResult& codeGenResult,
                            const std::string& resourcePathPrefix,
                            GiGlslShaderGen::MaterialGenInfo& genInfo)
  {
    // Append resource path prefix for file-backed MDL modules.
    std::vector<McTextureDescription> textureDescriptions = codeGenResult.textureDescriptions;

    if (!resourcePathPrefix.empty())
    {
      for (McTextureDescription& texRes : textureDescriptions)
      {
        texRes.filePath = resourcePathPrefix + texRes.filePath;
      }
    }

    // Remove MDL struct definitions because they're too bloated. We know more about the
    // data from which the code is generated from and can reduce the memory footprint.
    std::string glslSource = codeGenResult.source;
    size_t mdlCodeOffset = glslSource.find("// user defined structs");
    assert(mdlCodeOffset != std::string::npos);
    glslSource = glslSource.substr(mdlCodeOffset, glslSource.size() - mdlCodeOffset);

    genInfo = GiGlslShaderGen::MaterialGenInfo {
      .glslSource = glslSource,
      .textureDescriptions = textureDescriptions
    };

    return true;
  }

  bool _GenerateMaterialShadingGenInfo(McBackend& mcBackend,
                                       const McMaterial& material,
                                       GiGlslShaderGen::MaterialGenInfo& genInfo)
  {
    auto dfFlags = MC_DF_FLAG_SCATTERING | MC_DF_FLAG_VOLUME_ABSORPTION | MC_DF_FLAG_VOLUME_SCATTERING | MC_DF_FLAG_IOR;

    if (material.isEmissive)
    {
      dfFlags |= MC_DF_FLAG_EMISSION | MC_DF_FLAG_EMISSION_INTENSITY;
    }

    if (material.isThinWalled)
    {
      dfFlags |= MC_DF_FLAG_THIN_WALLED | MC_DF_FLAG_BACKFACE_SCATTERING;

      if (material.isEmissive)
      {
        dfFlags |= MC_DF_FLAG_BACKFACE_EMISSION | MC_DF_FLAG_BACKFACE_EMISSION_INTENSITY;
      }
    }

    McGlslGenResult genResult;
    if (!mcBackend.genGlsl(*material.mdlMaterial, McDfFlags(dfFlags), genResult))
    {
      return false;
    }

    return _MakeMaterialGenInfo(genResult, material.resourcePathPrefix, genInfo);
  }

  bool _GenerateMaterialOpacityGenInfo(McBackend& mcBackend,
                                       const McMaterial& material,
                                       GiGlslShaderGen::MaterialGenInfo& genInfo)
  {
    McDfFlags dfFlags = MC_DF_FLAG_CUTOUT_OPACITY;

    McGlslGenResult genResult;
    if (!mcBackend.genGlsl(*material.mdlMaterial, dfFlags, genResult))
    {
      return false;
    }

    return _MakeMaterialGenInfo(genResult, material.resourcePathPrefix, genInfo);
  }
}

namespace gtl
{
  GiShaderCacheFactory::GiShaderCacheFactory(CgpuDevice device,
                                             const CgpuPhysicalDeviceFeatures& deviceFeatures,
                                             GiGlslShaderGen& shaderGen,
                                             GiTextureManager& textureManager,
                                             McBackend& mcBackend)
    : m_device(device)
    , m_deviceFeatures(deviceFeatures)
    , m_shaderGen(shaderGen)
    , m_textureManager(textureManager)
    , m_mcBackend(mcBackend)
  {
  }

  GiShaderCache* GiShaderCacheFactory::create(const GiShaderCacheCreateInfo& createInfo)
  {
    bool clockCyclesAov = createInfo.aovId == GiAovId::ClockCycles;

    if (clockCyclesAov && !m_deviceFeatures.shaderClock)
    {
      GB_ERROR("unsupported AOV - device feature missing");
      return nullptr;
    }

    GB_LOG("material count: {}", createInfo.materialCount);
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

    uint32_t totalLightCount = createInfo.diskLightCount + createInfo.distantLightCount +
                               createInfo.rectLightCount + createInfo.sphereLightCount;

    bool nextEventEstimation = (createInfo.nextEventEstimation && totalLightCount > 0);

    GiGlslShaderGen::CommonShaderParams commonParams = {
      .aovId = (int) createInfo.aovId,
      .diskLightCount = createInfo.diskLightCount,
      .distantLightCount = createInfo.distantLightCount,
      .mediumStackSize = createInfo.mediumStackSize,
      .rectLightCount = createInfo.rectLightCount,
      .sphereLightCount = createInfo.sphereLightCount,
      .texCount2d = 2, // +1 fallback and +1 real dome light
      .texCount3d = 0
    };

    uint32_t& texCount2d = commonParams.texCount2d;
    uint32_t& texCount3d = commonParams.texCount3d;

    GiGlslDefines baseDefs;
    baseDefs.setDefine("AOV_ID", (int) createInfo.aovId);
    baseDefs.setDefine("TEXTURE_COUNT_2D", texCount2d);
    baseDefs.setDefine("TEXTURE_COUNT_3D", texCount3d);
    baseDefs.setDefine("SPHERE_LIGHT_COUNT", createInfo.sphereLightCount);
    baseDefs.setDefine("DISTANT_LIGHT_COUNT", createInfo.distantLightCount);
    baseDefs.setDefine("RECT_LIGHT_COUNT", createInfo.rectLightCount);
    baseDefs.setDefine("DISK_LIGHT_COUNT", createInfo.diskLightCount);
    baseDefs.setDefine("TOTAL_LIGHT_COUNT", totalLightCount);
    baseDefs.setDefine("MEDIUM_STACK_SIZE", createInfo.mediumStackSize);
#if defined(NDEBUG)
    baseDefs.setDefine("NDEBUG");
#endif

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
      hitGroupCompInfos.resize(createInfo.materialCount);

      std::atomic_bool threadWorkFailed = false;
#pragma omp parallel for
      for (int i = 0; i < int(hitGroupCompInfos.size()); i++)
      {
        const McMaterial* material = createInfo.materials[i];

        HitGroupCompInfo groupInfo;
        {
          GiGlslShaderGen::MaterialGenInfo genInfo;
	  if (!_GenerateMaterialShadingGenInfo(m_mcBackend, *material, genInfo))
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
          if (!_GenerateMaterialOpacityGenInfo(m_mcBackend, *material, genInfo))
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
        const McMaterial* material = createInfo.materials[i];

        HitGroupCompInfo& compInfo = hitGroupCompInfos[i];

        // Closest hit
        {
          GiGlslShaderGen::ClosestHitShaderParams hitParams = {
            .baseFileName = "rp_main.chit",
            .commonParams = commonParams,
            .directionalBias = material->directionalBias,
            .enableSceneTransforms = material->requiresSceneTransforms,
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

          if (!m_shaderGen.generateClosestHitSpirv(hitParams, compInfo.closestHitInfo.spv))
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
            .opacityEvalGlsl = compInfo.anyHitInfo->genInfo.glslSource,
            .textureIndexOffset2d = compInfo.anyHitInfo->texOffset2d,
            .textureIndexOffset3d = compInfo.anyHitInfo->texOffset3d
          };

          hitParams.shadowTest = false;
          if (!m_shaderGen.generateAnyHitSpirv(hitParams, compInfo.anyHitInfo->spv))
          {
            threadWorkFailed = true;
            continue;
          }

          hitParams.shadowTest = true;
          if (!m_shaderGen.generateAnyHitSpirv(hitParams, compInfo.anyHitInfo->shadowSpv))
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

            if (!cgpuCreateShader(m_device, {
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

            if (!cgpuCreateShader(m_device, {
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

            if (!cgpuCreateShader(m_device, {
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
      GiGlslDefines defs = baseDefs;
      defs.setConditionalDefine(createInfo.depthOfField, "DEPTH_OF_FIELD");
      defs.setConditionalDefine(createInfo.filterImportanceSampling, "FILTER_IMPORTANCE_SAMPLING");
      defs.setConditionalDefine(createInfo.nextEventEstimation, "NEXT_EVENT_ESTIMATION");
      defs.setConditionalDefine(createInfo.progressiveAccumulation, "PROGRESSIVE_ACCUMULATION");
      defs.setConditionalDefine(createInfo.depthOfField, "DEPTH_OF_FIELD");

      if (m_deviceFeatures.rayTracingInvocationReorder)
      {
        uint32_t reoderHintValueCount = createInfo.materialCount + 1/* no hit */;
        uint32_t reorderHintBitCount = 0;
  
        while (reoderHintValueCount >>= 1)
        {
          reorderHintBitCount++;
        }
  
        defs.setDefine("REORDER_INVOCATIONS");
        defs.setDefine("REORDER_HINT_BIT_COUNT", reorderHintBitCount);
      }

      rgenShader = m_shaderProvider.provide(GiShaderStage::RayGen, "rp_main.rgen", &defs);
      if (!rgenShader.handle)
        goto cleanup;
    }

    // Regular miss shader.
    {
      GiGlslDefines defs = baseDefs;
      defs.setConditionalDefine(createInfo.domeLightCameraVisible, "DOME_LIGHT_CAMERA_VISIBLE");

      CgpuShader shader = m_shaderProvider.provide(GiShaderStage::Miss, "rp_main.miss", &defs);
      if (!shader.handle)
        goto cleanup;

      missShaders.push_back(shader);
    }

    // Shadow miss shader.
    {
      CgpuShader shader = m_shaderProvider.provide(GiShaderStage::Miss, "rp_main_shadow.miss");
      if (!shader.handle)
        goto cleanup;

      missShaders.push_back(shader);
    }

    // Upload textures.
    if (textureDescriptions.size() > 0 && !m_textureManager.loadTextureDescriptions(textureDescriptions, images2d, images3d))
    {
      goto cleanup;
    }
    assert(images2d.size() == (texCount2d - 2));
    assert(images3d.size() == texCount3d);

    // Create RT pipeline.
    {
      GB_LOG("creating RT pipeline..");
      fflush(stdout);

      if (!cgpuCreateRtPipeline(m_device, {
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
    cache->aovId = (int) createInfo.aovId;
    cache->domeLightCameraVisible = createInfo.domeLightCameraVisible;
    cache->hitShaders = std::move(hitShaders);
    cache->images2d = std::move(images2d);
    cache->images3d = std::move(images3d);
    cache->materials.resize(createInfo.materialCount);
    for (uint32_t i = 0; i < createInfo.materialCount; i++)
    {
      cache->materials[i] = createInfo.materials[i];
    }
    cache->missShaders = missShaders;
    cache->pipeline = pipeline;
    cache->rgenShader = rgenShader;
    cache->hasPipelineClosestHitShader = hasPipelineClosestHitShader;
    cache->hasPipelineAnyHitShader = hasPipelineAnyHitShader;

cleanup:
    if (!cache)
    {
      m_textureManager.destroyUncachedImages(images2d);
      m_textureManager.destroyUncachedImages(images3d);
      if (rgenShader.handle)
      {
        cgpuDestroyShader(m_device, rgenShader);
      }
      for (CgpuShader shader : missShaders)
      {
        cgpuDestroyShader(m_device, shader);
      }
      for (CgpuShader shader : hitShaders)
      {
        cgpuDestroyShader(m_device, shader);
      }
      if (pipeline.handle)
      {
        cgpuDestroyPipeline(m_device, pipeline);
      }
    }
    return cache;

  }
}
