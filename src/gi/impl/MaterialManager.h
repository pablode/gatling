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

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <offsetAllocator.hpp>

#include <gtl/cgpu/Cgpu.h>
#include <gtl/gb/Hash.h>
#include <gtl/mc/Material.h>

#include "GlslShaderGen.h"
#include "TextureManager.h"
#include "interface/rp_main.h"

namespace gtl
{
  namespace rp = shader_interface::rp_main;

  class GbThreadPool;
  class GgpuDeleteQueue;
  class GgpuStager;
  class GiTextureManager;
  struct GiScene;

  struct GiMaterialGpuData
  {
    CgpuShader closestHit;
    std::vector<CgpuShader> anyHits; // optional
    OffsetAllocator::Allocation texOffsetAllocation;
    std::vector<GiImagePtr> images;
    std::vector<std::string> stringConstants;
    std::vector<rp::TextureInfo> textureInfos;
    std::vector<McTextureParameterBinding> textureParameterBindings;
    bool hasUdimTextures = false;
  };

  using GiMaterialGpuDataPtr = std::shared_ptr<GiMaterialGpuData>;

  // Per-material state owned by the material manager. It is embedded in GiMaterial, which is
  // defined by the renderer.
  struct GiBaseMaterial
  {
    McMaterial* mcMat = nullptr;
    std::string name;
    GbHash topoHash;
    GiMaterialGpuDataPtr gpuData;
    CgpuBuffer parameterBuffer;
    std::vector<uint8_t> parameterData;
    std::vector<McMtlxNetworkParameter> networkParameters;
    OffsetAllocator::Allocation textureParameterAllocation;
    std::vector<GiImagePtr> textureParameterImages;
  };

  struct GiImageBinding
  {
    GiImagePtr image;
    uint32_t index;
  };

  struct GiMaterialCompileResult
  {
    std::vector<GiImageBinding> imageBindings;
    std::vector<std::pair<uint32_t, rp::TextureInfo>> textureInfoEntries;
    uint32_t maxTextureIndex = 0;
  };

  class GiMaterialManager
  {
  public:
    GiMaterialManager(CgpuContext* ctx,
                      GiGlslShaderGen& shaderGen,
                      GiTextureManager& texSys,
                      GgpuStager& stager,
                      GgpuDeleteQueue& deleteQueue,
                      GbThreadPool& threadPool,
                      OffsetAllocator::Allocator& texAllocator);

    // Releases all GPU resources, including those of materials which are still alive.
    void destroy();

    // Adds the material to the manager.
    void addMaterial(GiBaseMaterial* material);

    // Removes the material from the manager. GPU resources are released on the render thread.
    void removeMaterial(GiBaseMaterial* material);

    std::vector<GiBaseMaterial*> materials();

    // To be called on the render thread once per frame.
    void housekeep();

    // Compiles all materials in parallel, deduplicating those with identical
    // topological hashes. Must be called on the render thread.
    bool compileMaterials(const std::vector<GiBaseMaterial*>& materials,
                          const GiGlslShaderGen::CommonShaderParams& commonParams,
                          uint32_t maxRayPayloadSize,
                          uint32_t maxRayHitAttributeSize,
                          bool nextEventEstimation,
                          bool needsAuxOutput,
                          bool disableDeduplication,
                          GiMaterialCompileResult& result);

  private:
    bool ensureParameterBuffers(const std::vector<GiBaseMaterial*>& materials);

    GiMaterialGpuDataPtr makeGpuDataPtr(GiMaterialGpuData&& gpuData);

  private:
    CgpuContext* m_ctx;
    GiGlslShaderGen& m_shaderGen;
    GiTextureManager& m_texSys;
    GgpuStager& m_stager;
    GgpuDeleteQueue& m_deleteQueue;
    GbThreadPool& m_threadPool;
    OffsetAllocator::Allocator& m_texAllocator;

    std::mutex m_mutex;
    std::unordered_set<GiBaseMaterial*> m_materials;
    std::vector<CgpuBuffer> m_pendingParameterBuffers;
    std::vector<GiMaterialGpuDataPtr> m_pendingGpuData;
    std::vector<std::vector<GiImagePtr>> m_pendingTextureParameterImages;
    std::vector<OffsetAllocator::Allocation> m_pendingTextureAllocations;
  };
}
