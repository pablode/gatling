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

#include "GlslShaderGen.h"
#include "GlslShaderCompiler.h"
#include "GlslStitcher.h"

#include <gtl/mc/Material.h>
#include <gtl/mc/Runtime.h>
#include <gtl/mc/Backend.h>

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <fstream>
#include <cassert>

namespace gtl
{
  class McRuntime;

  bool GiGlslShaderGen::init(std::string_view shaderPath, McRuntime& mcRuntime)
  {
    m_shaderPath = fs::path(shaderPath);

    m_mcBackend = std::make_shared<McBackend>();
    if (!m_mcBackend->init(mcRuntime))
    {
      return false;
    }

    m_shaderCompiler = std::make_shared<GiGlslShaderCompiler>(m_shaderPath);

    return true;
  }

  void _sgGenerateCommonDefines(GiGlslStitcher& stitcher, const GiGlslShaderGen::CommonShaderParams& params)
  {
#if defined(NDEBUG)
    stitcher.appendDefine("NDEBUG");
#endif
    stitcher.appendDefine("AOV_MASK", (int) params.aovMask);
    stitcher.appendDefine("MEDIUM_STACK_SIZE", (int32_t) params.mediumStackSize);
  }

  bool GiGlslShaderGen::generateRgenSpirv(std::string_view fileName, const RaygenShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    if (params.reorderInvocations)
    {
      uint32_t reoderHintValueCount = params.materialCount + 1/* no hit */;
      int32_t reorderHintBitCount = 0;

      while (reoderHintValueCount >>= 1)
      {
        reorderHintBitCount++;
      }

      stitcher.appendDefine("REORDER_INVOCATIONS");
      stitcher.appendDefine("REORDER_HINT_BIT_COUNT", reorderHintBitCount);
    }

    _sgGenerateCommonDefines(stitcher, params.commonParams);

    if (params.depthOfField)
    {
      stitcher.appendDefine("DEPTH_OF_FIELD");
    }
    if (params.filterImportanceSampling)
    {
      stitcher.appendDefine("FILTER_IMPORTANCE_SAMPLING");
    }
    if (params.jitteredSampling)
    {
      stitcher.appendDefine("JITTERED_SAMPLING");
    }
    if (params.nextEventEstimation)
    {
      stitcher.appendDefine("NEXT_EVENT_ESTIMATION");
    }
    if (params.progressiveAccumulation)
    {
      stitcher.appendDefine("PROGRESSIVE_ACCUMULATION");
    }
    if (params.clippingPlanes)
    {
      stitcher.appendDefine("CLIPPING_PLANES");
    }

    fs::path filePath = m_shaderPath / fileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiGlslShaderCompiler::ShaderStage::RayGen, source, spv);
  }

  bool GiGlslShaderGen::generateMissSpirv(std::string_view fileName, const MissShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.commonParams);

    if (params.domeLightCameraVisible)
    {
      stitcher.appendDefine("DOME_LIGHT_CAMERA_VISIBLE");
    }

    fs::path filePath = m_shaderPath / fileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiGlslShaderCompiler::ShaderStage::Miss, source, spv);
  }

  bool _MakeMaterialGenInfo(const McGlslGenResult& codeGenResult,
                            const std::string& resourcePathPrefix,
                            fs::path shaderPath,
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

  bool GiGlslShaderGen::generateMaterialInfo(const McMaterial& material, MaterialGenInfo& genInfo)
  {
    const std::unordered_map<McDf, const char*> dfs = {
      { McDf::Scattering, "mdl_bsdf_scattering" },
      { McDf::Emission, "mdl_edf_emission" },
      { McDf::EmissionIntensity, "mdl_edf_emission_intensity" },
      { McDf::ThinWalled, "mdl_thin_walled" },
      { McDf::VolumeAbsorption, "mdl_volume_absorption_coefficient" },
      { McDf::VolumeScattering, "mdl_volume_scattering_coefficient" },
      { McDf::CutoutOpacity, "mdl_cutout_opacity" },
      { McDf::Ior, "mdl_ior" },
      { McDf::BackfaceScattering, "mdl_backface_bsdf_scattering" },
      { McDf::BackfaceEmission, "mdl_backface_edf_emission" },
      { McDf::BackfaceEmissionIntensity, "mdl_backface_edf_emission_intensity" }
    };

    McGlslGenResult genResult;
    if (!m_mcBackend->genGlsl(*material.mdlMaterial, dfs, genResult))
    {
      return false;
    }

    return _MakeMaterialGenInfo(genResult, material.resourcePathPrefix, m_shaderPath, genInfo);
  }

  bool GiGlslShaderGen::generateClosestHitSpirv(const ClosestHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.commonParams);

    stitcher.appendDefine("TEXTURE_INDEX_OFFSET", (int32_t) params.textureIndexOffset);
    stitcher.appendDefine("MEDIUM_DIRECTIONAL_BIAS", params.directionalBias);
    stitcher.appendDefine("SCENE_DATA_COUNT", (int32_t) params.sceneDataCount);

    if (params.hasBackfaceBsdf)
    {
      stitcher.appendDefine("HAS_BACKFACE_BSDF");
    }
    if (params.hasBackfaceEdf)
    {
      stitcher.appendDefine("HAS_BACKFACE_EDF");
    }
    if (params.hasVolumeAbsorptionCoeff)
    {
      stitcher.appendDefine("HAS_VOLUME_ABSORPTION_COEFF");
    }
    if (params.hasVolumeScatteringCoeff)
    {
      stitcher.appendDefine("HAS_VOLUME_SCATTERING_COEFF");
    }
    if (params.isEmissive)
    {
      stitcher.appendDefine("IS_EMISSIVE");
    }
    if (params.hasCutoutTransparency)
    {
      stitcher.appendDefine("HAS_CUTOUT_TRANSPARENCY");
    }
    if (params.isThinWalled)
    {
      stitcher.appendDefine("IS_THIN_WALLED");
    }
    if (params.nextEventEstimation)
    {
      stitcher.appendDefine("NEXT_EVENT_ESTIMATION");
    }
    if (params.enableSceneTransforms)
    {
      stitcher.appendDefine("SCENE_TRANSFORMS");
    }
    if (params.cameraPositionSceneDataIndex > 0)
    {
      stitcher.appendDefine("CAMERA_POSITION_SCENE_DATA_INDEX", params.cameraPositionSceneDataIndex);
    }

    fs::path filePath = m_shaderPath / params.baseFileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    stitcher.replaceFirst("#pragma mdl_generated_code", params.shadingGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiGlslShaderCompiler::ShaderStage::ClosestHit, source, spv);
  }

  bool GiGlslShaderGen::generateAnyHitSpirv(const AnyHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.commonParams);

    stitcher.appendDefine("TEXTURE_INDEX_OFFSET", (int32_t) params.textureIndexOffset);
    stitcher.appendDefine("SCENE_DATA_COUNT", (int32_t) params.sceneDataCount);

    if (params.shadowTest)
    {
      stitcher.appendDefine("SHADOW_TEST");
    }
    if (params.enableSceneTransforms)
    {
      stitcher.appendDefine("SCENE_TRANSFORMS");
    }
    if (params.cameraPositionSceneDataIndex > 0)
    {
      stitcher.appendDefine("CAMERA_POSITION_SCENE_DATA_INDEX", params.cameraPositionSceneDataIndex);
    }

    fs::path filePath = m_shaderPath / params.baseFileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    stitcher.replaceFirst("#pragma mdl_generated_code", params.opacityEvalGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiGlslShaderCompiler::ShaderStage::AnyHit, source, spv);
  }

  bool GiGlslShaderGen::generateDenoisingSpirv(std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    fs::path filePath = m_shaderPath / "rp_denoise.comp";
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiGlslShaderCompiler::ShaderStage::Compute, source, spv);
  }
}
