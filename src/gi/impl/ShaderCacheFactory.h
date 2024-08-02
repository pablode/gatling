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

#pragma once

#include "Gi.h"
#include "GlslShaderProvider.h"

#include <gtl/cgpu/Cgpu.h>

namespace gtl
{
  class GiGlslShaderGen;
  class GiTextureManager;
  class McBackend;
  class McMaterial;

  struct GiShaderCache
  {
    uint32_t                       aovId = UINT32_MAX;
    bool                           domeLightCameraVisible;
    std::vector<CgpuShader>        hitShaders;
    std::vector<CgpuImage>         images2d;
    std::vector<CgpuImage>         images3d;
    std::vector<const McMaterial*> materials;
    std::vector<CgpuShader>        missShaders;
    CgpuPipeline                   pipeline;
    bool                           hasPipelineClosestHitShader = false;
    bool                           hasPipelineAnyHitShader = false;
    CgpuShader                     rgenShader;
    bool                           resetSampleOffset = true;
  };

  struct GiShaderCacheCreateInfo
  {
    GiAovId            aovId;
    bool               depthOfField;
    uint32_t           diskLightCount;
    uint32_t           distantLightCount;
    bool               domeLightCameraVisible;
    bool               filterImportanceSampling;
    const McMaterial** materials;
    uint32_t           materialCount;
    uint32_t           mediumStackSize;
    bool               nextEventEstimation;
    bool               progressiveAccumulation;
    uint32_t           rectLightCount;
    uint32_t           sphereLightCount;
  };

  class GiShaderCacheFactory
  {
  public:
    GiShaderCacheFactory(CgpuDevice device,
                         const CgpuPhysicalDeviceFeatures& deviceFeatures,
                         GiGlslShaderGen& shaderGen,
                         GiTextureManager& textureManager,
                         McBackend& mcBackend);

    GiShaderCache* create(const GiShaderCacheCreateInfo& createInfo);

  private:
    CgpuDevice m_device;
    const CgpuPhysicalDeviceFeatures& m_deviceFeatures;
    GiGlslShaderGen& m_shaderGen;
    GiTextureManager& m_textureManager;
    McBackend& m_mcBackend;
    GiGlslShaderProvider m_shaderProvider; // TODO: init
  };
}
