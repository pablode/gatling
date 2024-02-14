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

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>

#include <Backend.h>

namespace fs = std::filesystem;

namespace gtl
{
  struct McMaterial;
  class McRuntime;
  class McBackend;
  class GiGlslShaderCompiler;

  class GiGlslShaderGen
  {
  public:
    bool init(std::string_view shaderPath, McRuntime& runtime);

    ~GiGlslShaderGen();

  public:
    struct MaterialGenInfo
    {
      std::string glslSource;
      std::vector<McTextureDescription> textureDescriptions;
    };

    bool generateMaterialShadingGenInfo(const McMaterial& material, MaterialGenInfo& genInfo);
    bool generateMaterialOpacityGenInfo(const McMaterial& material, MaterialGenInfo& genInfo);

  public:
    struct RaygenShaderParams
    {
      int32_t aovId;
      bool depthOfField;
      bool filterImportanceSampling;
      uint32_t materialCount;
      bool nextEventEstimation;
      bool progressiveAccumulation;
      bool reorderInvocations;
      bool shaderClockExts;
      uint32_t distantLightCount;
      uint32_t sphereLightCount;
      uint32_t rectLightCount;
      uint32_t diskLightCount;
      uint32_t texCount2d;
      uint32_t texCount3d;
    };

    struct MissShaderParams
    {
      int32_t aovId;
      bool domeLightCameraVisible;
      uint32_t distantLightCount;
      uint32_t sphereLightCount;
      uint32_t rectLightCount;
      uint32_t diskLightCount;
      uint32_t texCount2d;
      uint32_t texCount3d;
    };

    struct ClosestHitShaderParams
    {
      int32_t aovId;
      std::string_view baseFileName;
      bool isOpaque;
      bool enableSceneTransforms;
      bool nextEventEstimation;
      std::string_view shadingGlsl;
      uint32_t distantLightCount;
      uint32_t sphereLightCount;
      uint32_t rectLightCount;
      uint32_t diskLightCount;
      uint32_t textureIndexOffset2d;
      uint32_t textureIndexOffset3d;
      uint32_t texCount2d;
      uint32_t texCount3d;
    };

    struct AnyHitShaderParams
    {
      int32_t aovId;
      bool enableSceneTransforms;
      std::string_view baseFileName;
      std::string_view opacityEvalGlsl;
      bool shadowTest;
      uint32_t distantLightCount;
      uint32_t sphereLightCount;
      uint32_t rectLightCount;
      uint32_t diskLightCount;
      uint32_t textureIndexOffset2d;
      uint32_t textureIndexOffset3d;
      uint32_t texCount2d;
      uint32_t texCount3d;
    };

    bool generateRgenSpirv(std::string_view fileName, const RaygenShaderParams& params, std::vector<uint8_t>& spv);
    bool generateMissSpirv(std::string_view fileName, const MissShaderParams& params, std::vector<uint8_t>& spv);
    bool generateClosestHitSpirv(const ClosestHitShaderParams& params, std::vector<uint8_t>& spv);
    bool generateAnyHitSpirv(const AnyHitShaderParams& params, std::vector<uint8_t>& spv);

  private:
    std::shared_ptr<McBackend> m_mcBackend;
    std::shared_ptr<GiGlslShaderCompiler> m_shaderCompiler;
    fs::path m_shaderPath;
  };
}
