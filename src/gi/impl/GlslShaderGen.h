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

#include <gtl/gb/Enum.h>
#include <gtl/mc/Backend.h>

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

  public:
    struct MaterialGenInfo
    {
      std::string glslSource;
      std::vector<McTextureDescription> textureDescriptions;
    };

    bool generateMaterialInfo(const McMaterial& material, MaterialGenInfo& genInfo);

  public:
    struct CommonShaderParams
    {
      uint32_t aovMask;
      uint32_t mediumStackSize;
    };

    struct RaygenShaderParams
    {
      bool clippingPlanes;
      CommonShaderParams commonParams;
      bool depthOfField;
      bool filterImportanceSampling;
      bool jitteredSampling;
      uint32_t materialCount;
      bool nextEventEstimation;
      bool progressiveAccumulation;
      bool reorderInvocations;
    };

    struct MissShaderParams
    {
      CommonShaderParams commonParams;
      bool domeLightCameraVisible;
    };

    struct ClosestHitShaderParams
    {
      std::string_view baseFileName;
      CommonShaderParams commonParams;
      float directionalBias;
      bool enableSceneTransforms;
      int cameraPositionSceneDataIndex;
      bool hasBackfaceBsdf;
      bool hasBackfaceEdf;
      bool hasCutoutTransparency;
      bool hasVolumeAbsorptionCoeff;
      bool hasVolumeScatteringCoeff;
      bool isEmissive;
      bool isThinWalled;
      bool nextEventEstimation;
      uint32_t sceneDataCount;
      std::string_view shadingGlsl;
      uint32_t textureIndexOffset;
    };

    struct AnyHitShaderParams
    {
      std::string_view baseFileName;
      CommonShaderParams commonParams;
      bool enableSceneTransforms;
      int cameraPositionSceneDataIndex;
      std::string_view opacityEvalGlsl;
      uint32_t sceneDataCount;
      bool shadowTest;
      uint32_t textureIndexOffset;
    };

    enum class OidnPostOp : uint32_t
    {
      None            = 0,
      MaxPool         = (1 << 0),
      Upsample        = (1 << 1),
      Concat          = (1 << 2),
      WriteBackRgba32 = (1 << 3),
      // TODO: this is not a post op. -> rename enum?
      ScaleInputInv   = (1 << 4),
      ScaleOutput     = (1 << 5)
// TODO: rename -> ScaleLuminance(Inv)
    };

    struct OidnParams
    {
      int wgSizeX;
      int wgSizeY;
      int in1ChannelCount;
      int in2ChannelCount = 0;
      int outChannelCount;
      int convChannelCount;
      int convolutionImpl;
      OidnPostOp postOp = OidnPostOp::None;
    };

    bool generateRgenSpirv(std::string_view fileName, const RaygenShaderParams& params, std::vector<uint8_t>& spv);
    bool generateMissSpirv(std::string_view fileName, const MissShaderParams& params, std::vector<uint8_t>& spv);
    bool generateClosestHitSpirv(const ClosestHitShaderParams& params, std::vector<uint8_t>& spv);
    bool generateAnyHitSpirv(const AnyHitShaderParams& params, std::vector<uint8_t>& spv);

    bool generateDenoisingSpirv(const OidnParams& params, std::vector<uint8_t>& spv);
    bool generateMaxLuminanceReductionSpirv(std::vector<uint8_t>& spv);

  private:
    std::shared_ptr<McBackend> m_mcBackend;
    std::shared_ptr<GiGlslShaderCompiler> m_shaderCompiler;
    fs::path m_shaderPath;
  };

  GB_DECLARE_ENUM_BITOPS(GiGlslShaderGen::OidnPostOp)
}
