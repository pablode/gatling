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

  bool GiGlslShaderGen::generateRgenSpirv(std::string_view fileName, const RaygenShaderParams& params, std::vector<uint8_t>& spv)
  {
  }

  bool GiGlslShaderGen::generateClosestHitSpirv(const ClosestHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    //_sgGenerateCommonDefines(stitcher, params.commonParams);

    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_2D", (int32_t) params.textureIndexOffset2d);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_3D", (int32_t) params.textureIndexOffset3d);
    stitcher.appendDefine("MEDIUM_DIRECTIONAL_BIAS", params.directionalBias);

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

    fs::path filePath = m_shaderPath / params.baseFileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    stitcher.replaceFirst("#pragma mdl_generated_code", params.shadingGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiShaderStage::ClosestHit, source, spv);
  }

  bool GiGlslShaderGen::generateAnyHitSpirv(const AnyHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GiGlslStitcher stitcher;
    stitcher.appendVersion();

    //_sgGenerateCommonDefines(stitcher, params.commonParams);

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

    stitcher.replaceFirst("#pragma mdl_generated_code", params.opacityEvalGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GiShaderStage::AnyHit, source, spv);
  }
}
