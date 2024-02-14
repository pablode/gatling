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

#include <Material.h>
#include <Runtime.h>
#include <Backend.h>

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

    if (!GiGlslShaderCompiler::init())
    {
      return false;
    }
    m_shaderCompiler = std::make_shared<GiGlslShaderCompiler>(m_shaderPath);

    return true;
  }

  GiGlslShaderGen::~GiGlslShaderGen()
  {
    GiGlslShaderCompiler::deinit();
  }

  void _sgGenerateCommonDefines(GiGlslStitcher& stitcher, uint32_t texCount2d, uint32_t texCount3d, uint32_t sphereLightCount,
                                uint32_t distantLightCount, uint32_t rectLightCount, uint32_t diskLightCount)
  {
#if defined(NDEBUG) || defined(__APPLE__)
    stitcher.appendDefine("NDEBUG");
#endif

    stitcher.appendDefine("TEXTURE_COUNT_2D", (int32_t) texCount2d);
    stitcher.appendDefine("TEXTURE_COUNT_3D", (int32_t) texCount3d);
    stitcher.appendDefine("SPHERE_LIGHT_COUNT", (int32_t) sphereLightCount);
    stitcher.appendDefine("DISTANT_LIGHT_COUNT", (int32_t) distantLightCount);
    stitcher.appendDefine("RECT_LIGHT_COUNT", (int32_t) rectLightCount);
    stitcher.appendDefine("DISK_LIGHT_COUNT", (int32_t) diskLightCount);
  }

  bool GiGlslShaderGen::generateRgenSpirv(std::string_view fileName, const RaygenShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    if (params.shaderClockExts)
    {
      stitcher.appendRequiredExtension("GL_EXT_shader_explicit_arithmetic_types_int64");
      stitcher.appendRequiredExtension("GL_ARB_shader_clock");
    }
    if (params.reorderInvocations)
    {
      stitcher.appendRequiredExtension("GL_NV_shader_invocation_reorder");
      // For hit shader invocation reordering hint
      stitcher.appendRequiredExtension("GL_EXT_buffer_reference");
      stitcher.appendRequiredExtension("GL_EXT_buffer_reference_uvec2");

      uint32_t reoderHintValueCount = params.materialCount + 1/* no hit */;
      int32_t reorderHintBitCount = 0;

      while (reoderHintValueCount >>= 1)
      {
        reorderHintBitCount++;
      }

      stitcher.appendDefine("REORDER_INVOCATIONS");
      stitcher.appendDefine("REORDER_HINT_BIT_COUNT", reorderHintBitCount);
    }

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d, params.sphereLightCount,
                             params.distantLightCount, params.rectLightCount, params.diskLightCount);

    if (params.depthOfField)
    {
      stitcher.appendDefine("DEPTH_OF_FIELD");
    }
    if (params.filterImportanceSampling)
    {
      stitcher.appendDefine("FILTER_IMPORTANCE_SAMPLING");
    }
    if (params.nextEventEstimation)
    {
      stitcher.appendDefine("NEXT_EVENT_ESTIMATION");
    }
    if (params.progressiveAccumulation)
    {
      stitcher.appendDefine("PROGRESSIVE_ACCUMULATION");
    }

    stitcher.appendDefine("AOV_ID", params.aovId);

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

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d, params.sphereLightCount,
                             params.distantLightCount, params.rectLightCount, params.diskLightCount);

    if (params.domeLightCameraVisible)
    {
      stitcher.appendDefine("DOME_LIGHT_CAMERA_VISIBLE");
    }

    stitcher.appendDefine("AOV_ID", params.aovId);

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

  bool GiGlslShaderGen::generateMaterialShadingGenInfo(const McMaterial& material, MaterialGenInfo& genInfo)
  {
    auto dfFlags = McDfFlags(MC_DF_FLAG_SCATTERING |
                             MC_DF_FLAG_EMISSION |
                             MC_DF_FLAG_EMISSION_INTENSITY |
                             MC_DF_FLAG_THIN_WALLED |
                             MC_DF_FLAG_VOLUME_ABSORPTION);

    McGlslGenResult genResult;
    if (!m_mcBackend->genGlsl(*material.mdlMaterial, dfFlags, genResult))
    {
      return false;
    }

    return _MakeMaterialGenInfo(genResult, material.resourcePathPrefix, m_shaderPath, genInfo);
  }

  bool GiGlslShaderGen::generateMaterialOpacityGenInfo(const McMaterial& material, MaterialGenInfo& genInfo)
  {
    McDfFlags dfFlags = MC_DF_FLAG_CUTOUT_OPACITY;

    McGlslGenResult genResult;
    if (!m_mcBackend->genGlsl(*material.mdlMaterial, dfFlags, genResult))
    {
      return false;
    }

    return _MakeMaterialGenInfo(genResult, material.resourcePathPrefix, m_shaderPath, genInfo);
  }

  bool GiGlslShaderGen::generateClosestHitSpirv(const ClosestHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d, params.sphereLightCount,
                             params.distantLightCount, params.rectLightCount, params.diskLightCount);

    stitcher.appendDefine("AOV_ID", params.aovId);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_2D", (int32_t) params.textureIndexOffset2d);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_3D", (int32_t) params.textureIndexOffset3d);
    if (params.isOpaque)
    {
      stitcher.appendDefine("IS_OPAQUE");
    }
    if (params.nextEventEstimation)
    {
      stitcher.appendDefine("NEXT_EVENT_ESTIMATION");
    }
    if (params.enableSceneTransforms)
    {
      stitcher.appendDefine("SCENE_TRANSFORMS");
    }

    fs::path filePath = m_shaderPath / params.baseFileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    stitcher.appendString(params.shadingGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiGlslShaderCompiler::ShaderStage::ClosestHit, source, spv);
  }

  bool GiGlslShaderGen::generateAnyHitSpirv(const AnyHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d, params.sphereLightCount,
                             params.distantLightCount, params.rectLightCount, params.diskLightCount);

    stitcher.appendDefine("AOV_ID", params.aovId);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_2D", (int32_t) params.textureIndexOffset2d);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_3D", (int32_t) params.textureIndexOffset3d);
    if (params.shadowTest)
    {
      stitcher.appendDefine("SHADOW_TEST");
    }
    if (params.enableSceneTransforms)
    {
      stitcher.appendDefine("SCENE_TRANSFORMS");
    }

    fs::path filePath = m_shaderPath / params.baseFileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    stitcher.appendString(params.opacityEvalGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiGlslShaderCompiler::ShaderStage::AnyHit, source, spv);
  }
}
